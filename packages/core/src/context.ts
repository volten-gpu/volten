/**
 * Options for creating a VoltenContext
 */
export interface VoltenOptions {
    /** Pre-existing GPUDevice to use. If not provided, one will be requested. */
    device?: GPUDevice;

    /** Pre-existing GPUAdapter to use. Only used if device is not provided. */
    adapter?: GPUAdapter;

    /** Label for debug purposes */
    label?: string;

    /**
     * Advanced override for uniform layout strategy.
     * - auto: use standard layout when extension is available, otherwise classic.
     * - classic: always use classic uniform constraints.
     * - standard: require uniform_buffer_standard_layout extension.
     */
    uniformLayoutMode?: 'auto' | 'classic' | 'standard';
}

/**
 * Pass-level options that override kernel defaults.
 */
export interface PassOptions {
    /**
     * Override thread dispatch count for this specific pass.
     * - number: Total 1D thread count
     * - [number]: Total 1D invocations
     * - [number, number]: Total 2D invocations (z defaults to 1)
     * - [number, number, number]: Total 3D invocations
     */
    threads?: number | [number] | [number, number] | [number, number, number];
}

import { Kernel } from './kernel/kernel.js';
import { PipelineCache } from './graph/pipeline-cache.js';
import {
    generateBindings,
    assembleFullShader,
    resolveBounds,
    resolveDispatch
} from './kernel/bindings.js';
import { VOLTEN_INTERNAL_BOUNDS_NAME } from './kernel/builtins.js';
import {
    createNode,
    getNodeOutputs,
    type Node,
    type Handle,
    isHandle
} from './graph/node.js';
import { compile, type ExecutionPlan } from './graph/compiler.js';
import { Buffer } from './data/buffer.js';
import { RawBuffer } from './data/raw-buffer.js';
import { Uniform } from './data/uniform.js';
import { getTypedArrayForType } from './utils/alignment.js';
import {
    resolveUniformLayoutMode,
    type UniformLayoutMode,
    type UniformLayoutPreference
} from './utils/uniform-layout.js';

/**
 * Valid targets for reading back data to the CPU.
 */
export type ReadTarget = Node | Buffer | RawBuffer | Handle;

const BOUNDS_GUARD_BARRIER_REGEX = /\b(?:workgroupBarrier|storageBarrier)\s*\(/;

function dispatchNeedsBoundsGuard(
    bounds: [number, number, number],
    dispatch: [number, number, number],
    workgroupSize: [number, number, number]
): boolean {
    return (
        dispatch[0] * workgroupSize[0] !== bounds[0] ||
        dispatch[1] * workgroupSize[1] !== bounds[1] ||
        dispatch[2] * workgroupSize[2] !== bounds[2]
    );
}

function kernelUsesBarrier(source: string): boolean {
    return BOUNDS_GUARD_BARRIER_REGEX.test(source);
}

/**
 * The main Volten context - the "v" instance
 *
 * This is created via the volten() factory function.
 * Contains methods for scheduling GPU compute work.
 */
export class VoltenContext {
    readonly device: GPUDevice;
    readonly label?: string;

    /** Pipeline cache for compute pipelines */
    readonly _pipelineCache: PipelineCache;
    /** Resolved uniform layout mode for this context */
    readonly _uniformLayoutMode: UniformLayoutMode;

    constructor(
        device: GPUDevice,
        options?: {
            label?: string;
            uniformLayoutMode?: UniformLayoutPreference;
        }
    ) {
        this.device = device;
        this.label = options?.label;
        this._pipelineCache = new PipelineCache();
        this._uniformLayoutMode = resolveUniformLayoutMode(
            options?.uniformLayoutMode
        );
    }

    /**
     * Create a compute pass node.
     *
     * This is the central API for building compute DAGs. It:
     * 1. Validates and classifies bindings (Buffer, RawBuffer, Uniform, Handle)
     * 2. Resolves logical invocation bounds and workgroup dispatch
     * 3. Generates WGSL binding declarations
     * 4. Assembles the full shader (bindings + kernel source)
     * 5. Creates or reuses a cached compute pipeline
     * 6. Returns a Node with output Handles for DAG chaining
     *
     * @param kernel - The kernel to execute
     * @param bindings - Input/output bindings (Buffer, RawBuffer, Uniform, or Handle from previous pass)
     * @param options - Optional pass configuration (e.g., thread override)
     * @returns A Node handle for chaining or execution
     *
     * @example
     * ```ts
     * const input = new Buffer([1, 2, 3, 4], "f32");
     * const output = new Buffer([0, 0, 0, 0], "f32", "rw");
     *
     * const A = v.pass(new Kernel(`
     *   fn main(gid: vec3u) {
     *     output[gid.x] = input[gid.x] * 2.0;
     *   }
     * `, { threads: 'input' }), { input, output });
     *
     * // Chain passes:
     * const B = v.pass(AnotherKernel, { data: A.output, result: resultBuf });
     * ```
     */
    pass(
        kernel: Kernel,
        bindings: Record<string, Buffer | RawBuffer | Uniform | Handle> = {},
        options?: PassOptions
    ): Node {
        // 1. Validate kernel type
        if (!(kernel instanceof Kernel)) {
            throw new Error(
                'Volten Error: First argument to v.pass() must be a Kernel instance.\n' +
                    '  Example: v.pass(new Kernel(`fn main(gid: vec3u) { ... }`), { ... })'
            );
        }

        if (VOLTEN_INTERNAL_BOUNDS_NAME in bindings) {
            throw new Error(
                `Volten Error: Binding name "${VOLTEN_INTERNAL_BOUNDS_NAME}" is reserved for internal use.`
            );
        }

        // 2. Resolve logical bounds and physical dispatch dimensions
        const bounds = resolveBounds(kernel, bindings, options?.threads);
        const dispatch = resolveDispatch(kernel, bindings, options?.threads);

        if (
            !kernel.unsafeManualBounds &&
            kernelUsesBarrier(kernel.source) &&
            dispatchNeedsBoundsGuard(bounds, dispatch, kernel.workgroupSize)
        ) {
            throw new Error(
                'Volten Error: This kernel uses workgroup/storage barriers and also requires a guarded partial workgroup.\n' +
                    '  An injected early-return wrapper would be unsafe here.\n' +
                    '  Fix one of these:\n' +
                    '  1. Make threads an exact multiple of workgroupSize\n' +
                    '  2. Set unsafeManualBounds: true and handle bounds manually in WGSL'
            );
        }

        const executionBindings = kernel.unsafeManualBounds
            ? bindings
            : {
                  ...bindings,
                  [VOLTEN_INTERNAL_BOUNDS_NAME]: new Uniform(
                      [bounds[0], bounds[1], bounds[2], 0],
                      'vec4u'
                  )
              };

        // 3. Generate binding entries (classify & validate)
        const bindingEntries = generateBindings(executionBindings, kernel, {
            uniformLayoutMode: this._uniformLayoutMode
        });

        // 4. Assemble full shader source
        const shaderCode = assembleFullShader(kernel, bindingEntries, {
            uniformLayoutMode: this._uniformLayoutMode
        });

        // 5. Get or create pipeline
        const { pipeline, bindGroupLayout } = this._pipelineCache.getOrCreate(
            this.device,
            shaderCode
        );

        // 6. Collect dependencies (nodes that provide Handle inputs)
        const dependencies: Node[] = [];
        const seen = new Set<symbol>();
        for (const value of Object.values(bindings)) {
            if (isHandle(value) && !seen.has(value._node._id)) {
                seen.add(value._node._id);
                dependencies.push(value._node);
            }
        }

        // 7. Create and return the Node
        return createNode({
            kernel,
            pipeline,
            bindGroupLayout,
            bindingEntries,
            bounds: [...bounds],
            dispatch: [...dispatch],
            bindings,
            shaderCode,
            dependencies
        });
    }

    /**
     * Resolve a binding value to its concrete Buffer or RawBuffer.
     * - Buffer / RawBuffer → returns self
     * - Handle → recursively walks up the Handle chain to find the actual buffer
     */
    private _resolveConcreteBuffer(
        value: Buffer | RawBuffer | Handle
    ): Buffer | RawBuffer {
        if (value instanceof Buffer || value instanceof RawBuffer) {
            return value;
        }
        if (isHandle(value)) {
            // Recursively resolve: the source node's binding for this name
            // may itself be a Handle (multi-level chaining)
            const sourceBinding = value._node._bindings[value._name];
            return this._resolveConcreteBuffer(
                sourceBinding as Buffer | RawBuffer | Handle
            );
        }
        throw new Error(
            `Volten Error: Cannot resolve concrete buffer from binding of type ${typeof value}.`
        );
    }

    /**
     * Resolve a binding value to its GPUBuffer.
     */
    private _resolveGPUBuffer(
        value: Buffer | RawBuffer | Uniform | Handle
    ): GPUBuffer {
        if (value instanceof Uniform) {
            return value.ensure(this.device);
        }
        return this._resolveConcreteBuffer(value).ensure(this.device);
    }

    // -----------------------------------------------------------------------
    // Compile → Submit pipeline
    // -----------------------------------------------------------------------

    /**
     * Compile one or more terminal nodes into an ExecutionPlan.
     *
     * This is a pure graph analysis step — no GPU calls. It:
     * - Merges multiple terminal subtrees
     * - Detects shared concrete buffers between independent subtrees
     * - Injects synthetic dependencies for buffer-overlapping nodes
     * - Produces a topologically sorted execution order
     *
     * This is the seam where pool allocation will slot in (future).
     *
     * @param terminals - Terminal nodes in the order specified by v.run()
     * @returns An ExecutionPlan with nodes in safe execution order
     */
    private _compile(terminals: Node[]): ExecutionPlan {
        return compile(terminals);
    }

    /**
     * Submit an ExecutionPlan to the GPU.
     *
     * Takes a pre-compiled plan (sorted node list) and:
     * 1. Ensures all GPU buffers are uploaded
     * 2. Creates bind groups (resolving Handles to actual GPUBuffers)
     * 3. Manages compute passes (reuses when possible, breaks on hazards)
     * 4. Encodes dispatches
     * 5. Submits the command buffer
     *
     * @param plan - The compiled execution plan
     */
    private _submit(plan: ExecutionPlan): void {
        const encoder = this.device.createCommandEncoder();

        // We'll try to re-use the pass & current pipeline
        // as much as possible without beginning new compute passes
        let pass: GPUComputePassEncoder | null = null;
        let currentPipeline: GPUComputePipeline | null = null;

        // Track nodes that have been executed within the *current* pass.
        // If a subsequent node depends on any of these, we must end the pass
        // to ensure memory visibility rules (Read-After-Write hazard).
        const nodesInCurrentPass = new Set<Node>();

        for (const node of plan.sorted) {
            // Ensure all Buffer/RawBuffer bindings are uploaded
            for (const entry of node._bindingEntries) {
                if (
                    entry.source instanceof Buffer ||
                    entry.source instanceof RawBuffer ||
                    entry.source instanceof Uniform
                ) {
                    entry.source.ensure(this.device);
                }
            }

            // Build bind group entries, resolving Handles to actual GPUBuffers
            const bgEntries: GPUBindGroupEntry[] = node._bindingEntries.map(
                (entry) => ({
                    binding: entry.index,
                    resource: {
                        buffer: this._resolveGPUBuffer(entry.source)
                    }
                })
            );

            const bindGroup = this.device.createBindGroup({
                layout: node._bindGroupLayout,
                entries: bgEntries
            });

            // Check for dependencies within the current pass
            let dependsOnCurrentPass = false;
            if (pass) {
                for (const dep of node._dependencies) {
                    if (nodesInCurrentPass.has(dep)) {
                        dependsOnCurrentPass = true;
                        break;
                    }
                }
            }

            // If we have a hazard (dependency in current pass) or no active pass, start a new one
            if (!pass || dependsOnCurrentPass) {
                if (pass) {
                    pass.end();
                    nodesInCurrentPass.clear();
                    currentPipeline = null;
                }
                pass = encoder.beginComputePass();
            }

            // Set pipeline only if changed
            if (currentPipeline !== node._pipeline) {
                pass.setPipeline(node._pipeline);
                currentPipeline = node._pipeline;
            }

            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(
                node._dispatch[0],
                node._dispatch[1],
                node._dispatch[2]
            );

            // Mark this node as executed in the current pass
            nodesInCurrentPass.add(node);
        }

        if (pass) {
            pass.end();
        }

        this.device.queue.submit([encoder.finish()]);
    }

    /**
     * Execute one or more terminal nodes (fire and forget).
     * Does not wait for GPU completion.
     *
     * When multiple terminals are provided, the positional order
     * determines scheduling priority for independent nodes that share
     * concrete buffers. Handle-connected nodes are always ordered
     * correctly regardless of argument order.
     *
     * @param node - Terminal node(s) to execute
     */
    run(node: Node | Node[]): void {
        const terminals = Array.isArray(node) ? node : [node];
        const plan = this._compile(terminals);
        this._submit(plan);
    }

    /**
     * Execute one or more terminal nodes and wait for GPU completion.
     *
     * @param node - Terminal node(s) to execute
     */
    async wait(node: Node | Node[]): Promise<void> {
        const terminals = Array.isArray(node) ? node : [node];
        const plan = this._compile(terminals);
        this._submit(plan);
        await this.device.queue.onSubmittedWorkDone();
    }

    /**
     * Read back targets (Nodes, Buffers, Handles) from GPU to CPU.
     *
     * If passed a single target, returns data for that target.
     * If passed an array of targets, returns an array of results.
     *
     * A Node target returns a Record of its output names to their ArrayBuffer/TypedArray.
     * A Buffer/RawBuffer/Handle returns a single ArrayBuffer/TypedArray.
     */
    async read(target: ReadTarget): Promise<any>;
    async read(targets: ReadTarget[]): Promise<any[]>;
    async read(target: ReadTarget | ReadTarget[]): Promise<any> {
        const isArrayTarget = Array.isArray(target);
        const targets = isArrayTarget ? target : [target];

        // 1. Determine all unique concrete buffers we need to read
        // this is necessary to prevent duplicate requests
        // Mapping from Concrete Buffer -> { staging: GPUBuffer, size: number }
        const uniqueBuffers = new Map<
            Buffer | RawBuffer,
            { staging: GPUBuffer; size: number }
        >();

        // once we retrieve all the buffers, we'll iterate through these "plans" to map
        // the resolved arrays to the appropriate return targets
        type TargetPlan =
            | {
                  type: 'node';
                  node: Node;
                  outputs: { name: string; concrete: Buffer | RawBuffer }[];
              }
            | { type: 'buffer'; concrete: Buffer | RawBuffer };

        const targetPlans: TargetPlan[] = [];

        for (const t of targets) {
            if (t instanceof Buffer || t instanceof RawBuffer || isHandle(t)) {
                const concrete = this._resolveConcreteBuffer(t);
                uniqueBuffers.set(concrete, {
                    staging: null!,
                    size: concrete.byteLength
                });
                targetPlans.push({ type: 'buffer', concrete });
            } else {
                // It's a Node
                const nodeOutputs = getNodeOutputs(t as Node);
                const outputNames = Object.keys(nodeOutputs);
                if (outputNames.length === 0) {
                    throw new Error(
                        'Volten Error: v.read() called on a node with no declared outputs.\n' +
                            '  Hint: Make sure your Kernel has outputs declared, e.g.:\n' +
                            "    new Kernel(`...`, { outputs: ['result'] })"
                    );
                }
                const outputs: {
                    name: string;
                    concrete: Buffer | RawBuffer;
                }[] = [];
                for (const name of outputNames) {
                    const handle = nodeOutputs[name];
                    const concrete = this._resolveConcreteBuffer(handle);
                    uniqueBuffers.set(concrete, {
                        staging: null!,
                        size: concrete.byteLength
                    });
                    outputs.push({ name, concrete });
                }
                targetPlans.push({ type: 'node', node: t as Node, outputs });
            }
        }

        if (uniqueBuffers.size === 0) {
            return isArrayTarget ? [] : undefined;
        }

        const encoder = this.device.createCommandEncoder();

        // 2. Create staging buffers and issue copies
        for (const [concrete, info] of uniqueBuffers.entries()) {
            const gpuBuffer = concrete.ensure(this.device);
            const staging = this.device.createBuffer({
                size: info.size,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
            });
            info.staging = staging;
            encoder.copyBufferToBuffer(gpuBuffer, 0, staging, 0, info.size);
        }

        this.device.queue.submit([encoder.finish()]);

        // 3. Map memory and extract TypedArrays
        const materializedBuffers = new Map<
            Buffer | RawBuffer,
            ArrayBufferView | ArrayBuffer
        >();

        for (const [concrete, info] of uniqueBuffers.entries()) {
            await info.staging.mapAsync(GPUMapMode.READ);
            const mappedRange = info.staging.getMappedRange().slice(0); // Clone the data securely

            let TypedArrayConstructor:
                | Float32ArrayConstructor
                | Uint32ArrayConstructor
                | Int32ArrayConstructor
                | undefined = Float32Array;
            if (concrete instanceof Buffer) {
                TypedArrayConstructor = getTypedArrayForType(concrete.type);
            }

            let data: ArrayBufferView | ArrayBuffer;
            if (TypedArrayConstructor) {
                data = new TypedArrayConstructor(mappedRange);
            } else {
                data = mappedRange; // ArrayBuffer for structs or raw data
            }

            info.staging.unmap();
            info.staging.destroy();

            materializedBuffers.set(concrete, data);
        }

        // 4. Reconstruct results
        const results = targetPlans.map((plan) => {
            if (plan.type === 'buffer') {
                return materializedBuffers.get(plan.concrete)!;
            } else {
                const record: Record<string, ArrayBufferView | ArrayBuffer> =
                    {};
                for (const out of plan.outputs) {
                    record[out.name] = materializedBuffers.get(out.concrete)!;
                }
                return record;
            }
        });

        return isArrayTarget ? results : results[0];
    }
}

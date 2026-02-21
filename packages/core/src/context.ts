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
}

/**
 * Pass-level options that override kernel defaults.
 */
export interface PassOptions {
    /**
     * Override thread dispatch count for this specific pass.
     * - number: Total 1D thread count
     * - [number, number]: Total 2D invocations (z defaults to 1)
     * - [number, number, number]: Total 3D invocations
     */
    threads?: number | [number, number] | [number, number, number];
}

import { Kernel } from './kernel/kernel.js';
import { PipelineCache } from './graph/pipeline-cache.js';
import {
    generateBindings,
    assembleFullShader,
    resolveDispatch,
} from './kernel/bindings.js';
import {
    createNode,
    getNodeOutputs,
    type Node,
    type Handle,
    isHandle,
} from './graph/node.js';
import { compile, type ExecutionPlan } from './graph/compiler.js';
import { Buffer } from './data/buffer.js';
import { RawBuffer } from './data/raw-buffer.js';
import { getTypedArrayForType } from './utils/alignment.js';

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

    constructor(device: GPUDevice, options?: { label?: string }) {
        this.device = device;
        this.label = options?.label;
        this._pipelineCache = new PipelineCache();
    }

    /**
     * Create a compute pass node.
     * 
     * This is the central API for building compute DAGs. It:
     * 1. Validates and classifies bindings (Buffer, RawBuffer, Handle)
     * 2. Generates WGSL binding declarations
     * 3. Assembles the full shader (bindings + kernel source)
     * 4. Creates or reuses a cached compute pipeline
     * 5. Resolves thread dispatch dimensions
     * 6. Returns a Node with output Handles for DAG chaining
     * 
     * @param kernel - The kernel to execute
     * @param bindings - Input/output bindings (Buffer, RawBuffer, or Handle from previous pass)
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
     * `, { outputs: ['output'], threads: 'input' }), { input, output });
     * 
     * // Chain passes:
     * const B = v.pass(AnotherKernel, { data: A.output, result: resultBuf });
     * ```
     */
    pass(
        kernel: Kernel,
        bindings: Record<string, Buffer | RawBuffer | Handle> = {},
        options?: PassOptions
    ): Node {
        // 1. Validate kernel type
        if (!(kernel instanceof Kernel)) {
            throw new Error(
                'Volten Error: First argument to v.pass() must be a Kernel instance.\n' +
                '  Example: v.pass(new Kernel(`fn main(gid: vec3u) { ... }`), { ... })'
            );
        }

        // 2. Generate binding entries (classify & validate)
        const bindingEntries = generateBindings(bindings, kernel);

        // 3. Assemble full shader source
        const shaderCode = assembleFullShader(kernel, bindingEntries);

        // 4. Get or create pipeline
        const { pipeline, bindGroupLayout } = this._pipelineCache.getOrCreate(
            this.device,
            shaderCode
        );

        // 5. Resolve dispatch dimensions
        const dispatch = resolveDispatch(
            kernel,
            bindings,
            options?.threads
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
            dispatch: [...dispatch],
            bindings,
            shaderCode,
            dependencies,
        });
    }

    /**
     * Resolve a binding value to its GPUBuffer.
     * - Buffer / RawBuffer → call ensure(device)
     * - Handle → recursively walk up the Handle chain to find the actual buffer
     */
    private _resolveGPUBuffer(value: Buffer | RawBuffer | Handle): GPUBuffer {
        if (value instanceof Buffer || value instanceof RawBuffer) {
            return value.ensure(this.device);
        }
        if (isHandle(value)) {
            // Recursively resolve: the source node's binding for this name
            // may itself be a Handle (multi-level chaining)
            const sourceBinding = value._node._bindings[value._name];
            return this._resolveGPUBuffer(sourceBinding as Buffer | RawBuffer | Handle);
        }
        throw new Error(
            `Volten Error: Cannot resolve GPU buffer from binding of type ${typeof value}.`
        );
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
                if (entry.source instanceof Buffer || entry.source instanceof RawBuffer) {
                    entry.source.ensure(this.device);
                }
            }

            // Build bind group entries, resolving Handles to actual GPUBuffers
            const bgEntries: GPUBindGroupEntry[] = node._bindingEntries.map((entry) => ({
                binding: entry.index,
                resource: {
                    buffer: this._resolveGPUBuffer(entry.source),
                },
            }));

            const bindGroup = this.device.createBindGroup({
                layout: node._bindGroupLayout,
                entries: bgEntries,
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
                node._dispatch[2],
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
     *
     * When multiple terminals are provided, the positional order
     * determines scheduling priority for independent nodes that share
     * concrete buffers. Handle-connected nodes are always ordered
     * correctly regardless of argument order.
     *
     * @param nodes - Terminal node(s) to execute
     */
    run(node: Node): void;
    run(...nodes: Node[]): void;
    run(nodes: Node[]): void;
    run(...args: (Node | Node[])[]): void {
        const terminals = this._normalizeTerminals(args);
        const plan = this._compile(terminals);
        this._submit(plan);
    }

    /**
     * Execute one or more terminal nodes and wait for GPU completion.
     *
     * @param nodes - Terminal node(s) to execute
     */
    async wait(node: Node): Promise<void>;
    async wait(...nodes: Node[]): Promise<void>;
    async wait(nodes: Node[]): Promise<void>;
    async wait(...args: (Node | Node[])[]): Promise<void> {
        const terminals = this._normalizeTerminals(args);
        const plan = this._compile(terminals);
        this._submit(plan);
        await this.device.queue.onSubmittedWorkDone();
    }

    /**
     * Read back a node's output buffers to CPU.
     *
     * Returns a record mapping output names to their typed arrays.
     *
     * @param node - The terminal node to read from
     * @returns CPU-readable data (typed array or record of typed arrays)
     */
    async read(node: Node): Promise<Record<string, ArrayBufferView | ArrayBuffer>> {
        const outputs = getNodeOutputs(node);
        const outputNames = Object.keys(outputs);

        if (outputNames.length === 0) {
            throw new Error(
                'Volten Error: v.read() called on a node with no declared outputs.\n' +
                '  Hint: Make sure your Kernel has outputs declared, e.g.:\n' +
                '    new Kernel(`...`, { outputs: [\'result\'] })'
            );
        }

        // For each output, create a staging buffer, copy, and read back
        const result: Record<string, ArrayBufferView | ArrayBuffer> = {};
        const encoder = this.device.createCommandEncoder();
        const stagingBuffers: { name: string; staging: GPUBuffer; size: number; binding: Buffer | RawBuffer | Handle }[] = [];

        for (const name of outputNames) {
            const binding = node._bindings[name];
            if (!(binding instanceof Buffer) && !(binding instanceof RawBuffer)) {
                throw new Error(
                    `Volten Error: Output "${name}" is not a Buffer/RawBuffer — cannot read back.`
                );
            }

            const gpuBuffer = binding.ensure(this.device);
            const size = binding.byteLength;

            const staging = this.device.createBuffer({
                size,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });

            encoder.copyBufferToBuffer(gpuBuffer, 0, staging, 0, size);
            stagingBuffers.push({ name, staging, size, binding });
        }

        this.device.queue.submit([encoder.finish()]);

        // Map all staging buffers and read data
        for (const { name, staging, size, binding } of stagingBuffers) {
            await staging.mapAsync(GPUMapMode.READ);
            const mappedRange = staging.getMappedRange().slice(0);

            let TypedArrayConstructor: Float32ArrayConstructor | Uint32ArrayConstructor | Int32ArrayConstructor | undefined = Float32Array;
            if (binding instanceof Buffer) {
                TypedArrayConstructor = getTypedArrayForType(binding.type);
            }

            let data: ArrayBufferView | ArrayBuffer;
            if (TypedArrayConstructor) {
                data = new TypedArrayConstructor(mappedRange);
            } else {
                data = mappedRange; // ArrayBuffer for structs or unsupported
            }

            staging.unmap();
            staging.destroy();
            result[name] = data;
        }

        return result;
    }

    /**
     * Normalize the variadic arguments from run/wait into a flat Node array.
     * Supports: run(node), run(A, B, C), run([A, B, C])
     */
    private _normalizeTerminals(args: (Node | Node[])[]): Node[] {
        // run([A, B, C]) — single array argument
        if (args.length === 1 && Array.isArray(args[0])) {
            return args[0] as Node[];
        }
        // run(A) or run(A, B, C) — individual node arguments
        return args as Node[];
    }
}

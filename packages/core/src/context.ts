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

import { PipelineCache } from './graph/pipeline-cache.js';
import { getOrCreateNodeBindGroup } from './kernel/bind-groups.js';
import { resolveConcreteBuffer } from './kernel/resource-resolution.js';
import { type Node, type Handle, isHandle, isNode } from './graph/node.js';
import type { DispatchHandle, DispatchNode } from './graph/dispatch-node.js';
import { Materializer } from './graph/materializer.js';
import { compile, type ExecutionPlan } from './graph/compiler.js';
import { collectNodesFromMultiple } from './graph/scheduler.js';
import { Buffer } from './data/buffer.js';
import { RawBuffer } from './data/raw-buffer.js';
import { getTypedArrayForType } from './utils/alignment.js';
import {
    resolveUniformLayoutMode,
    type UniformLayoutMode,
    type UniformLayoutPreference
} from './utils/uniform-layout.js';
import { decodeDebugBuffer, type DebugReadResult } from './debug/index.js';

export type { InvocationOptions } from './graph/node.js';

/**
 * Valid targets for reading back data to the CPU.
 */
export type ReadTarget = Node | Buffer | RawBuffer | Handle;

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
    private readonly _materializer: Materializer;

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
        this._materializer = new Materializer({
            device,
            pipelineCache: this._pipelineCache,
            uniformLayoutMode: this._uniformLayoutMode
        });
    }

    // -----------------------------------------------------------------------
    // Compile → Submit pipeline
    // -----------------------------------------------------------------------

    /**
     * Materialize logical nodes, then put their dispatches in a safe order.
     *
     * Materialization may select a context-specific shader and create its
     * pipeline. The compiler then handles shared buffers and sorts the
     * physical dispatch graph.
     */
    private _compile(nodes: readonly Node[]): ExecutionPlan | null {
        const invocations = this._materializer
            .materialize(nodes)
            .filter((invocation) => invocation.terminals.length > 0);
        return invocations.length > 0 ? compile(invocations) : null;
    }

    /**
     * Submit an ExecutionPlan to the GPU.
     *
     * Takes a pre-compiled plan (sorted node list) and:
     * 1. Ensures all GPU buffers are uploaded
     * 2. Gets or creates bind groups (resolving Handles to actual GPUBuffers)
     * 3. Manages compute passes (reuses when possible, breaks on hazards)
     * 4. Encodes dispatches
     * 5. Submits the command buffer
     *
     * @param plan - The compiled execution plan
     */
    private _submit(plan: ExecutionPlan): void {
        const encoder = this.device.createCommandEncoder({
            label: this.label
                ? `${this.label} command encoder`
                : 'Volten command encoder'
        });

        // We'll try to re-use the pass & current pipeline
        // as much as possible without beginning new compute passes
        let pass: GPUComputePassEncoder | null = null;
        let currentPipeline: GPUComputePipeline | null = null;

        // Track nodes that have been executed within the *current* pass.
        // If a subsequent node depends on any of these, we must end the pass
        // to ensure memory visibility rules (Read-After-Write hazard).
        const nodesInCurrentPass = new Set<DispatchNode>();

        for (const node of plan.sorted) {
            if (node._debug) {
                node._debug.resource.reset(this.device);
            }

            const bindGroup = getOrCreateNodeBindGroup(this.device, node);

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
                pass = encoder.beginComputePass({
                    label: `Volten compute pass starting at ${node._label}`
                });
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
        const nodes = Array.isArray(node) ? node : [node];
        const execution = this._compile(nodes);
        if (execution) this._submit(execution);
    }

    /**
     * Execute one or more terminal nodes and wait for GPU completion.
     *
     * @param node - Terminal node(s) to execute
     */
    async wait(node: Node | Node[]): Promise<void> {
        const nodes = Array.isArray(node) ? node : [node];
        const execution = this._compile(nodes);
        if (execution) this._submit(execution);
        await this.device.queue.onSubmittedWorkDone();
    }

    /**
     * Release Volten-owned internal resources for one or more nodes.
     *
     * This walks the reachable dependency subtree and destroys only resources
     * created by Volten itself (for example hidden bounds uniforms). User
     * buffers and user-provided uniforms are never touched.
     *
     * Calling v.destroy(node) multiple times is safe.
     *
     * @param node - Terminal node(s) whose internal subtree resources should be released
     */
    destroy(node: Node | Node[]): void {
        const nodes = Array.isArray(node) ? node : [node];
        const terminals = nodes.flatMap((current) => [
            ...(this._materializer.getCached(current)?.terminals ?? [])
        ]);
        const reachable = collectNodesFromMultiple(terminals);

        for (const current of reachable) {
            for (const resource of current._ownedResources) {
                resource.destroy();
            }
        }
    }

    private async _readConcreteBuffers(
        concretes: readonly (Buffer | RawBuffer)[]
    ): Promise<Map<Buffer | RawBuffer, ArrayBufferView | ArrayBuffer>> {
        const uniqueBuffers = new Map<
            Buffer | RawBuffer,
            { staging: GPUBuffer; size: number }
        >();

        for (const concrete of concretes) {
            uniqueBuffers.set(concrete, {
                staging: null!,
                size: concrete.byteLength
            });
        }

        if (uniqueBuffers.size === 0) {
            return new Map();
        }

        const encoder = this.device.createCommandEncoder({
            label: this.label
                ? `${this.label} readback encoder`
                : 'Volten readback encoder'
        });

        for (const [concrete, info] of uniqueBuffers.entries()) {
            const gpuBuffer = concrete.ensure(this.device);
            const staging = this.device.createBuffer({
                label: `${concrete.label} readback`,
                size: info.size,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
            });
            info.staging = staging;
            encoder.copyBufferToBuffer(gpuBuffer, 0, staging, 0, info.size);
        }

        this.device.queue.submit([encoder.finish()]);

        const materializedBuffers = new Map<
            Buffer | RawBuffer,
            ArrayBufferView | ArrayBuffer
        >();

        for (const [concrete, info] of uniqueBuffers.entries()) {
            await info.staging.mapAsync(GPUMapMode.READ);
            const mappedRange = info.staging.getMappedRange().slice(0);

            let TypedArrayConstructor:
                | Float32ArrayConstructor
                | Uint32ArrayConstructor
                | Int32ArrayConstructor
                | undefined;
            if (concrete instanceof Buffer) {
                TypedArrayConstructor = getTypedArrayForType(concrete.type);
            }

            let data: ArrayBufferView | ArrayBuffer;
            if (TypedArrayConstructor) {
                data = new TypedArrayConstructor(mappedRange);
            } else {
                data = mappedRange;
            }

            info.staging.unmap();
            info.staging.destroy();

            materializedBuffers.set(concrete, data);
        }

        return materializedBuffers;
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
                  outputs: { name: string; concrete: Buffer | RawBuffer }[];
              }
            | { type: 'buffer'; concrete: Buffer | RawBuffer };

        const targetPlans: TargetPlan[] = [];

        for (const t of targets) {
            if (t instanceof Buffer || t instanceof RawBuffer || isHandle(t)) {
                const concrete = this._resolveReadBuffer(t);
                uniqueBuffers.set(concrete, {
                    staging: null!,
                    size: concrete.byteLength
                });
                targetPlans.push({ type: 'buffer', concrete });
            } else if (isNode(t)) {
                const lowered = this._materializer.lowerNode(t);
                const outputNames = lowered.outputNames;
                if (outputNames.length === 0) {
                    throw new Error(
                        'Volten Error: v.read() called on a node with no declared outputs.\n' +
                            '  A kernel declares readable outputs with kernel({ outputs: [...] }); a plan returns them from its builder.'
                    );
                }
                const outputs: {
                    name: string;
                    concrete: Buffer | RawBuffer;
                }[] = [];
                for (const name of outputNames) {
                    const resolved = lowered.outputs.get(name);
                    if (!resolved) {
                        throw new Error(
                            `Volten Error: Operation "${t._label}" declares output "${name}" but does not expose a matching buffer handle.`
                        );
                    }
                    const concrete = this._resolveReadBuffer(resolved.resource);
                    uniqueBuffers.set(concrete, {
                        staging: null!,
                        size: concrete.byteLength
                    });
                    outputs.push({ name, concrete });
                }
                targetPlans.push({ type: 'node', outputs });
            } else {
                throw new Error('Volten Error: Invalid v.read() target.');
            }
        }

        if (uniqueBuffers.size === 0) {
            return isArrayTarget ? [] : undefined;
        }
        const materializedBuffers = await this._readConcreteBuffers(
            Array.from(uniqueBuffers.keys())
        );

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

    async readDebug(node: Node): Promise<DebugReadResult> {
        if (node._operation._kind === 'plan') {
            throw new Error(
                'Volten Error: v.readDebug() does not support plan nodes yet.'
            );
        }

        const lowered = this._materializer.lowerNode(node);
        const dispatch = lowered.terminals[0];
        if (!dispatch?._debug) {
            throw new Error(
                'Volten Error: v.readDebug() called on a node without debug support.\n' +
                    '  Hint: Enable it when invoking the kernel: operation(bindings, { debug: true }).'
            );
        }

        const readback = await this._readConcreteBuffers([
            dispatch._debug.resource.buffer
        ]);
        const raw = readback.get(dispatch._debug.resource.buffer);
        if (!(raw instanceof ArrayBuffer)) {
            throw new Error(
                'Volten Error: Internal debug readback expected an ArrayBuffer payload.'
            );
        }

        return decodeDebugBuffer(
            raw,
            dispatch._debug.messages,
            dispatch._debug.resource.bufferSize
        );
    }

    private _resolveReadBuffer(
        value: Buffer | RawBuffer | Handle | DispatchHandle
    ): Buffer | RawBuffer {
        const physical = isHandle(value)
            ? this._materializer.resolveHandle(value).resource
            : value;
        const concrete = resolveConcreteBuffer(physical);
        if (concrete) return concrete;
        throw new Error(
            'Volten Error: Cannot resolve a concrete buffer from the read target.'
        );
    }
}

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
import { createNode, type Node, type Handle, isHandle } from './graph/node.js';
import { Buffer } from './data/buffer.js';
import { RawBuffer } from './data/raw-buffer.js';

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
     * Execute a node or chain of nodes (fire and forget)
     * Does not wait for GPU completion.
     * 
     * @param node - The node to execute
     */
    run(node: unknown): void {
        // TODO: Implement in next phase
        throw new Error('v.run() not yet implemented');
    }

    /**
     * Execute a node and wait for GPU completion
     * 
     * @param node - The node to execute
     */
    async wait(node: unknown): Promise<void> {
        // TODO: Implement in next phase
        throw new Error('v.wait() not yet implemented');
    }

    /**
     * Execute a node and read back results to CPU
     * 
     * @param node - The node to read from
     * @returns The CPU-readable data
     */
    async read(node: unknown): Promise<unknown> {
        // TODO: Implement in next phase
        throw new Error('v.read() not yet implemented');
    }
}



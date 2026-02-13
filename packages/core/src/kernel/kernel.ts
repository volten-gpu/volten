// Kernel class
// Stores shader source code, output declarations, and thread configuration

import { processShaderSource } from './builtins.js';

/** Output size specification for pool allocation */
export type OutputSize = number | ((data: Record<string, unknown>) => number);

/** Configuration for a single output */
export interface OutputConfig {
    /** 
     * Output size specification. If omitted, defaults to same size as primary input.
     * - number: Fixed size
     * - function: Dynamic size based on pass-time inputs
     */
    size?: OutputSize;
}

/** Output declarations - simple array or detailed config */
export type OutputsSpec = string[] | Record<string, OutputConfig>;

/**
 * Specifies how many GPU threads to launch for a kernel.
 *
 * All forms express total invocations. Volten divides each axis by
 * the corresponding workgroup dimension (with ceil) to compute dispatch size.
 *
 * - **number:** 1D invocation count, equivalent to [N, 1, 1].
 * - **[x, y] or [x, y, z]:** Per-axis invocation counts.
 * - **string:** Name of a binding whose element count determines the count (1D).
 * - **function:** Dynamic — return number, [x,y], or [x,y,z].
 *
 * @example
 * threads: 1024                           // → dispatch [ceil(1024/wgX), 1, 1]
 * threads: [512, 512]                     // → dispatch [ceil(512/wgX), ceil(512/wgY), 1]
 * threads: 'input'                        // → infer count from binding named "input"
 * threads: (data) => data.input.count     // → dynamic 1D
 */
export type ThreadsSpec =
    | number                                                    // Simple 1D count
    | string                                                    // Infer from named input
    | ((data: Record<string, unknown>) => number | [number, number] | [number, number, number]); // Dynamic

/**
 * Options for Kernel creation
 */
export interface KernelOptions {
    /** 
     * Output declarations.
     * - string[]: Shorthand, all outputs same size as primary input
     * - Record<string, OutputConfig>: Explicit size control per output
     */
    outputs?: OutputsSpec;

    /** 
     * Workgroup size for the compute shader.
     * Defaults to [64, 1, 1].
     * ^ This default is not optimal for kernels that work in 2D spaces and
     *   require frequent neighbor lookups like blur / stencil ops, I decided
     *   however to keep this as an initial default and handle the optimal
     *   workgroup size at the stdlib level, e.g. the author of a gaussianBlur 
     *   kernel should use the correct workgroup size for that type of kernel.
     *   Workgroup size is a compile-time, algorithmic concern. 
     *   Dispatch dimensions are a runtime, data-shape concern
     * The shader will have @workgroup_size(x, y, z) injected.
     */
    workgroupSize?: [number, number?, number?];

    /**
     * Thread dispatch configuration.
     * Controls how many total invocations to dispatch.
     * - number: Fixed 1D dispatch count
     * - string: Infer from the length of the named input buffer
     * - function: Compute dynamically from pass-time inputs
     * 
     * If omitted, Volten will attempt to infer from a single input.
     */
    threads?: ThreadsSpec;
}

/**
 * Normalized output format (internal use)
 */
export interface NormalizedOutput {
    name: string;
    size?: OutputSize;
}

/**
 * Kernel class for compute shader definitions.
 * 
 * Handles:
 * - Builtin shorthand expansion (gid, lid, wid, lid3, nwg)
 * - @compute and @workgroup_size injection
 * - Output declarations with optional size specifications
 * - Thread dispatch configuration
 * 
 * @example
 * ```ts
 * // Simple kernel with shorthand builtins
 * const MyKernel = new Kernel(`
 *   fn main(gid: vec3<u32>) {
 *     output[gid.x] = input[gid.x] * 2.0;
 *   }
 * `, { outputs: ['output'] });
 * 
 * // Kernel with explicit output size (e.g., reduce)
 * const ReduceKernel = new Kernel(`...`, {
 *   outputs: { result: { size: 1 } },
 *   workgroupSize: [256],
 * });
 * ```
 */
export class Kernel {
    /** The original user-provided WGSL source */
    readonly source: string;

    /** Normalized output declarations */
    readonly outputs: NormalizedOutput[];

    /** Workgroup size as [x, y, z] */
    readonly workgroupSize: [number, number, number];

    /** Thread dispatch specification */
    readonly threads?: ThreadsSpec;

    /** Cached assembled source (with injections) */
    private _assembledSource: string | null = null;

    constructor(
        source: string,
        options?: KernelOptions
    ) {
        this.source = source;
        this.outputs = this.normalizeOutputs(options?.outputs);
        this.workgroupSize = this.normalizeWorkgroupSize(options?.workgroupSize);
        this.threads = options?.threads;
    }

    /**
     * Get the assembled WGSL source with all transformations applied:
     * - Builtin shorthand expansion
     * - @compute and @workgroup_size injection
     * 
     * Note: Binding injections (group/binding) are deferred to the DAG compiler
     * at v.pass() time when actual buffers are known.
     */
    get assembledSource(): string {
        if (this._assembledSource === null) {
            this._assembledSource = processShaderSource(
                this.source,
                this.workgroupSize
            );
        }
        return this._assembledSource;
    }

    /**
     * Get output names as a simple array
     */
    get outputNames(): string[] {
        return this.outputs.map(o => o.name);
    }

    /**
     * Normalize outputs from either string[] or Record<string, OutputConfig>
     */
    private normalizeOutputs(spec?: OutputsSpec): NormalizedOutput[] {
        if (!spec) return [];

        if (Array.isArray(spec)) {
            // Simple array of names
            return spec.map(name => ({ name }));
        }

        // Record<string, OutputConfig>
        return Object.entries(spec).map(([name, config]) => ({
            name,
            size: config.size,
        }));
    }

    /**
     * Normalize workgroup size to [x, y, z] format
     */
    private normalizeWorkgroupSize(
        spec?: [number, number?, number?]
    ): [number, number, number] {
        if (!spec) return [64, 1, 1];
        const [x, y = 1, z = 1] = spec;
        return [x, y, z];
    }
}

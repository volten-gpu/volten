// Kernel class
// Stores shader source code, output declarations, and thread configuration

import { makeLabel } from '../utils/labels.js';

const BARRIER_USAGE_REGEX = /\b(?:workgroupBarrier|storageBarrier)\s*\(/;

/** Output size specification for pool allocation */
export type OutputSize = number | ((data: Record<string, unknown>) => number);

/** Configuration for a single output */
export interface OutputConfig {
    /**
     * Infer type (and optionally size) from another binding.
     * The named binding's wgslType and element count are copied.
     * Resolved at v.pass() time when actual bindings are known.
     */
    definedBy?: string;

    /**
     * Explicit WGSL type override (e.g. "array<vec3f>").
     * Takes precedence over definedBy for type.
     */
    type?: string;

    /**
     * Output size specification (element count).
     * Takes precedence over definedBy for size.
     * - number: Fixed size
     * - function: Dynamic size based on pass-time inputs
     */
    size?: OutputSize;
}

/**
 * Output declarations.
 *
 * - string[]: semantic output names only. Useful when users provide buffers.
 * - Record<string, OutputConfig>: output names plus shape metadata for future
 *   Volten-owned auto-allocation.
 */
export type OutputsSpec = readonly string[] | Record<string, OutputConfig>;

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
    | number // Simple 1D count
    | string // Infer from named input
    | ((
          data: Record<string, unknown>
      ) => number | [number, number] | [number, number, number]); // Dynamic

/**
 * Options for Kernel creation
 */
export interface KernelOptions {
    /** Optional human-friendly label for debugger/devtools usage. */
    label?: string;

    /**
     * Output declarations for pool-allocation preparation.
     *
     * Record<string, OutputConfig>: declarative output description with
     * definedBy / type / size fields.
     *
     * @example
     * ```ts
     * outputs: {
     *   output: { definedBy: 'input' },        // copy type & size from "input"
     *   result: { type: 'array<f32>', size: 1 }, // explicit override
     *   half:   { definedBy: 'input', size: (d) => (d.input as Buffer).count / 2 },
     * }
     * ```
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

    /**
     * Opt out of Volten's hidden dispatch-bounds guard.
     *
     * Use this for advanced kernels that manage bounds manually, or for
     * workgroup-cooperative kernels where an injected early return would be
     * inappropriate.
     */
    unsafeManualBounds?: boolean;
}

/**
 * Normalized output format (internal use)
 */
export interface NormalizedOutput {
    name: string;
    definedBy?: string;
    type?: string;
    size?: OutputSize;
}

/**
 * Kernel class for compute shader definitions.
 *
 * Handles:
 * - Builtin shorthand expansion (gid, lid, wid, lid3, nwg)
 * - Guarded entry-point generation by default
 * - @compute and @workgroup_size injection
 * - Output declarations with optional size specifications
 * - Thread dispatch configuration
 *
 * @example
 * ```ts
 * // Simple kernel — no outputs needed for in-place work
 * const InPlace = new Kernel(`
 *   fn main(gid: vec3<u32>) {
 *     data[gid.x] *= 2.0;
 *   }
 * `);
 *
 * // Kernel with declarative outputs (for pool allocation)
 * const BlurKernel = new Kernel(`...`, {
 *   outputs: { output: { definedBy: 'input' } },
 * });
 *
 * // Kernel with explicit output size (e.g., reduce)
 * const ReduceKernel = new Kernel(`...`, {
 *   outputs: { result: { type: 'array<f32>', size: 1 } },
 *   workgroupSize: [256],
 * });
 * ```
 */
export class Kernel {
    /** Human-friendly debug label */
    readonly label: string;

    /** The original user-provided WGSL source */
    readonly source: string;

    /**
     * Normalized output declarations, object form { name, definedBy?, type?, size? }
     */
    readonly outputs: NormalizedOutput[];

    /** Workgroup size as [x, y, z] */
    readonly workgroupSize: [number, number, number];

    /** Thread dispatch specification */
    readonly threads?: ThreadsSpec;

    /** Whether the kernel source uses WGSL synchronization barriers. */
    readonly usesBarrier: boolean;

    /** Skip Volten's hidden dispatch-bounds guard and manage bounds manually. */
    readonly unsafeManualBounds: boolean;

    constructor(source: string, options?: KernelOptions) {
        this.label = makeLabel('Kernel', options?.label);
        this.source = source;
        this.outputs = this.normalizeOutputs(options?.outputs);
        this.workgroupSize = this.normalizeWorkgroupSize(
            options?.workgroupSize
        );
        this.threads = options?.threads;
        this.usesBarrier = BARRIER_USAGE_REGEX.test(source);
        this.unsafeManualBounds = options?.unsafeManualBounds ?? false;
    }

    /**
     * Get output names as a simple array
     */
    get outputNames(): string[] {
        return this.outputs.map((o) => o.name);
    }

    /**
     * Normalize outputs from Record<string, OutputConfig> to NormalizedOutput[]
     */
    private normalizeOutputs(spec?: OutputsSpec): NormalizedOutput[] {
        if (!spec) return [];

        if (Array.isArray(spec)) {
            return spec.map((name) => ({
                name,
                definedBy: undefined,
                type: undefined,
                size: undefined
            }));
        }

        return Object.entries(spec).map(([name, config]) => ({
            name,
            definedBy: config.definedBy,
            type: config.type,
            size: config.size
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

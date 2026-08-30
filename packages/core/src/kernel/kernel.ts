import {
    createLogicalNode,
    type Bindings,
    type HandlesForBindings,
    type InvocationOptions,
    type Node,
    type Handle
} from '../graph/node.js';
import type {
    OperationContext,
    OperationDefinition
} from '../graph/operation.js';
import { makeLabel } from '../utils/labels.js';

const BARRIER_USAGE_REGEX = /\b(?:workgroupBarrier|storageBarrier)\s*\(/;

/**
 * Output element-count metadata.
 *
 * A fixed number describes a statically sized output. A function can derive
 * the size from the bindings supplied when the callable kernel is invoked.
 * This is retained as allocation metadata; callers currently still provide
 * the concrete output buffer themselves.
 */
export type OutputSize = number | ((data: Record<string, unknown>) => number);

/** Declarative metadata for one named kernel output. */
export interface OutputConfig {
    /**
     * Binding whose type and element count describe this output.
     * Explicit `type` and `size` values take precedence over the corresponding
     * values implied by `definedBy`.
     */
    definedBy?: string;

    /** Explicit WGSL type, for example `array<vec3f>`. */
    type?: string;

    /**
     * Explicit output element count, either fixed or derived from invocation
     * bindings. This overrides the count implied by `definedBy`.
     */
    size?: OutputSize;
}

/**
 * Declares the public outputs of a kernel.
 *
 * Use an array when only the binding names matter:
 *
 *     outputs: ['result', 'debug']
 *
 * Use an object when an output also carries shape metadata:
 *
 *     outputs: {
 *         result: { definedBy: 'input' },
 *         total: { type: 'array<f32>', size: 1 }
 *     }
 *
 * In both forms, every name must match a buffer-like invocation binding.
 * Declared names determine whole-node readback and are excluded when Volten
 * looks for an input buffer from which to infer a thread count.
 */
export type OutputsSpec = readonly string[] | Record<string, OutputConfig>;

/**
 * Describes the total logical invocations launched by a kernel.
 *
 * - `number`: fixed 1D count, equivalent to `[count, 1, 1]`.
 * - `string`: infer the 1D count from the named Buffer binding.
 * - `function`: derive a 1D, 2D, or 3D count from invocation bindings.
 *
 * These are invocation counts, not workgroup counts. Volten divides each axis
 * by `workgroupSize` with `ceil` to obtain the WebGPU dispatch dimensions.
 *
 * @example
 * ```ts
 * threads: 1024
 * threads: 'input'
 * threads: ({ image }) => [image.width, image.height]
 * ```
 */
export type ThreadsSpec =
    | number
    | string
    | ((
          data: Record<string, unknown>
      ) => number | [number, number] | [number, number, number]);

/**
 * WGSL source selected either statically or once a GPU context is available.
 * A selector is evaluated during materialization, not while authoring the
 * logical graph, so it may safely inspect device features and limits.
 */
export type KernelShader = string | ((context: OperationContext) => string);

/** Declarative input accepted by `kernel()`. */
export interface KernelConfig {
    /** WGSL kernel source, or a context-dependent source selector. */
    shader: KernelShader;

    /** Optional human-friendly label used by diagnostics and GPU tooling. */
    label?: string;

    /**
     * Binding names exposed as kernel outputs. Besides controlling whole-node
     * readback, these names help distinguish outputs during thread inference.
     */
    outputs?: OutputsSpec;

    /**
     * Compile-time workgroup dimensions injected as
     * `@workgroup_size(x, y, z)`. Missing axes default to `1`; the complete
     * default is `[64, 1, 1]`.
     *
     * Workgroup size is an algorithmic choice: neighbor-heavy 2D kernels, for
     * example, generally need a shape chosen for that algorithm. `threads`, by
     * contrast, describes the runtime data shape to cover.
     */
    workgroupSize?: [number, number?, number?];

    /**
     * Total logical invocation count. If omitted, Volten infers it from the
     * only unambiguous input Buffer; invocation options may override it.
     */
    threads?: ThreadsSpec;

    /**
     * Disable Volten's injected dispatch-bounds guard.
     *
     * Use this only when the WGSL performs its own bounds handling, or for
     * workgroup-cooperative code where an injected early return would make
     * barriers unsafe. The default is `false`.
     */
    unsafeManualBounds?: boolean;
}

/** Options used by the concrete, already-resolved kernel representation. */
export type KernelOptions = Omit<KernelConfig, 'shader'>;

/** Object form used internally after either `OutputsSpec` form is normalized. */
export interface NormalizedOutput {
    readonly name: string;
    readonly definedBy?: string;
    readonly type?: string;
    readonly size?: OutputSize;
}

/**
 * Immutable kernel metadata retained by the callable operation.
 * GPU-specific work deliberately does not happen here; shader selection,
 * binding classification, and pipeline creation happen during v.run().
 */
export class KernelDefinition {
    readonly label: string;
    readonly shader: KernelShader;
    readonly outputs: readonly NormalizedOutput[];
    readonly workgroupSize: [number, number, number];
    readonly threads?: ThreadsSpec;
    readonly unsafeManualBounds: boolean;

    constructor(config: KernelConfig) {
        this.label = makeLabel('Kernel', config.label);
        this.shader = config.shader;
        this.outputs = normalizeOutputs(config.outputs);
        this.workgroupSize = normalizeWorkgroupSize(config.workgroupSize);
        this.threads = config.threads;
        this.unsafeManualBounds = config.unsafeManualBounds ?? false;
    }

    /** Names exposed by logical nodes and by whole-node readback. */
    get outputNames(): string[] {
        return this.outputs.map((output) => output.name);
    }

    /** Select the shader and produce context-specific materialization data. */
    resolve(context: OperationContext): ResolvedKernel {
        const source =
            typeof this.shader === 'function'
                ? this.shader(context)
                : this.shader;

        return {
            label: this.label,
            source,
            outputs: this.outputs,
            outputNames: this.outputNames,
            workgroupSize: this.workgroupSize,
            threads: this.threads,
            usesBarrier: BARRIER_USAGE_REGEX.test(source),
            unsafeManualBounds: this.unsafeManualBounds
        };
    }
}

/** Concrete kernel metadata used by the existing WGSL and dispatch pipeline. */
export interface ResolvedKernel {
    /** Human-friendly label retained from the definition. */
    readonly label: string;
    /** Concrete WGSL selected for this GPU context. */
    readonly source: string;
    /** Normalized output declarations. */
    readonly outputs: readonly NormalizedOutput[];
    /** Output binding names in declaration order. */
    readonly outputNames: readonly string[];
    /** Fully normalized `[x, y, z]` workgroup dimensions. */
    readonly workgroupSize: [number, number, number];
    /** Kernel-level logical invocation rule, before invocation overrides. */
    readonly threads?: ThreadsSpec;
    /** Whether the selected WGSL contains a synchronization barrier call. */
    readonly usesBarrier: boolean;
    /** Whether Volten must leave bounds handling entirely to the WGSL. */
    readonly unsafeManualBounds: boolean;
}

/**
 * Concrete kernel metadata consumed by WGSL helpers and physical dispatches.
 * The public authoring API is kernel({ shader, ... }); this class remains an
 * internal value object for code that already has a resolved shader string.
 */
export class Kernel implements ResolvedKernel {
    readonly label: string;
    readonly source: string;
    readonly outputs: readonly NormalizedOutput[];
    readonly workgroupSize: [number, number, number];
    readonly threads?: ThreadsSpec;
    readonly usesBarrier: boolean;
    readonly unsafeManualBounds: boolean;

    constructor(source: string, options: KernelOptions = {}) {
        this.label = makeLabel('Kernel', options.label);
        this.source = source;
        this.outputs = normalizeOutputs(options.outputs);
        this.workgroupSize = normalizeWorkgroupSize(options.workgroupSize);
        this.threads = options.threads;
        this.usesBarrier = BARRIER_USAGE_REGEX.test(source);
        this.unsafeManualBounds = options.unsafeManualBounds ?? false;
    }

    get outputNames(): string[] {
        return this.outputs.map((output) => output.name);
    }
}

type OutputNames<TConfig extends KernelConfig> =
    TConfig['outputs'] extends readonly (infer TName extends string)[]
        ? TName
        : TConfig['outputs'] extends Record<string, OutputConfig>
          ? keyof TConfig['outputs'] & string
          : never;

type KernelNode<TBindings extends Bindings, TOutput extends string> = Node<
    HandlesForBindings<TBindings> & Record<TOutput, Handle>
>;

export interface KernelOperation<
    TOutput extends string = never
> extends OperationDefinition {
    readonly _kind: 'kernel';
    readonly _definition: KernelDefinition;
    readonly outputNames: readonly TOutput[];

    <TBindings extends Bindings>(
        bindings: TBindings,
        options?: InvocationOptions
    ): KernelNode<TBindings, TOutput>;
}

/**
 * Defines a callable primitive GPU operation.
 *
 * Calling the returned function only records bindings in a logical node:
 *
 *     const add = kernel({ shader: `...`, threads: 'lhs' });
 *     const A = add({ lhs, rhs, result });
 *
 * The shader may inspect the eventual device through OperationContext, but it
 * is not evaluated until A is materialized by a Volten context.
 */
export function kernel<const TConfig extends KernelConfig>(
    config: TConfig
): KernelOperation<OutputNames<TConfig>> {
    const definition = new KernelDefinition(config);

    const operation = ((bindings: Bindings, options?: InvocationOptions) =>
        createLogicalNode(operation, bindings, options)) as KernelOperation<
        OutputNames<TConfig>
    >;

    Object.defineProperties(operation, {
        _kind: { value: 'kernel' },
        _definition: { value: definition },
        label: { value: definition.label },
        outputNames: { value: definition.outputNames }
    });

    return operation;
}

export function isKernelOperation(
    operation: OperationDefinition
): operation is KernelOperation<string> {
    return operation._kind === 'kernel';
}

function normalizeOutputs(spec?: OutputsSpec): readonly NormalizedOutput[] {
    if (!spec) return [];
    if (Array.isArray(spec)) {
        return spec.map((name) => ({ name }));
    }
    return Object.entries(spec).map(([name, config]) => ({
        name,
        definedBy: config.definedBy,
        type: config.type,
        size: config.size
    }));
}

/** Expand omitted workgroup axes and apply Volten's 1D default. */
function normalizeWorkgroupSize(
    spec?: [number, number?, number?]
): [number, number, number] {
    if (!spec) return [64, 1, 1];
    const [x, y = 1, z = 1] = spec;
    return [x, y, z];
}

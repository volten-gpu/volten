// Node and Handle types
// Opaque handles for DAG nodes and their outputs

import type { Kernel } from '../kernel/kernel.js';
import type { BindingEntry } from '../kernel/bindings.js';
import { Buffer } from '../data/buffer.js';
import { RawBuffer } from '../data/raw-buffer.js';
import type { NodeDebugState } from '../debug/resource.js';

/**
 * Handle to a buffer output from a compute pass.
 * Opaque type - users shouldn't inspect internal structure.
 */
export interface Handle {
    /** Internal identifier - prefixed with _ to denote internal property */
    readonly _id: symbol;
    /** Reference to parent node */
    readonly _node: Node;
    /** Output name for debugging/identification */
    readonly _name: string;
    /** Human-friendly debug label */
    readonly _label: string;
}

/**
 * Type guard for Handle.
 * Handles are objects with _id (symbol), _node, and _name properties.
 */
export function isHandle(value: unknown): value is Handle {
    return (
        typeof value === 'object' &&
        value !== null &&
        '_id' in value &&
        '_node' in value &&
        '_name' in value &&
        '_label' in value &&
        typeof (value as Handle)._id === 'symbol'
    );
}

/**
 * Check if a value is a buffer-like binding (Buffer, RawBuffer, or Handle).
 * These are the binding types that should generate handles on a Node.
 */
export function isBufferLike(
    value: unknown
): value is Buffer | RawBuffer | Handle {
    return (
        value instanceof Buffer || value instanceof RawBuffer || isHandle(value)
    );
}

// -----------------------------------------------------------------------------
// Node Design Philosophy:
// -----------------------------------------------------------------------------
// Output handles are spread directly onto the Node for API terseness:
//
//   const A = v.pass(K1, { in: input });
//   const B = v.pass(K2, { in: A.output });
//
// Convention:
// - Internal properties are prefixed with "_" (e.g., _id, _dependencies)
// - User-facing output names must NOT start with "_" (validated at runtime)
// - This separation allows iteration over outputs via Object.keys filtering
//
// All buffer-like bindings (Buffer, RawBuffer, Handle) get handles on the Node.
// This enables in-place buffer re-use without
// declaring outputs:
//
//   const A = v.pass(K, { inout: buf });
//   const B = v.pass(K, { inout: A.inout });
// -----------------------------------------------------------------------------

/** Reserved property names that cannot be used as output names */
export const RESERVED_NODE_PROPERTIES = [
    '_id',
    '_dependencies',
    '_kernel',
    '_pipeline',
    '_bindGroupLayout',
    '_bindingEntries',
    '_ownedResources',
    '_bounds',
    '_dispatch',
    '_bindings',
    '_shaderCode',
    '_label',
    '_cachedBindGroup',
    '_debug'
] as const;

/**
 * Internal GPU-backed resource owned by a node.
 *
 * These resources are created by Volten itself (not by the user) and may be
 * explicitly released via v.destroy(node) without affecting user-managed
 * buffers or uniforms.
 */
export interface NodeOwnedResource {
    destroy(): void;
}

export interface CachedNodeBindGroup {
    readonly key: string;
    readonly bindGroup: GPUBindGroup;
}

/**
 * Validates that an output name doesn't conflict with internal properties.
 * @throws Error if the name starts with "_" or is a reserved property name
 */
export function validateOutputName(name: string): void {
    if (name.startsWith('_')) {
        throw new Error(
            `Volten Error: Output name "${name}" cannot start with "_". ` +
                `Names starting with "_" are reserved for internal properties.`
        );
    }
    if (RESERVED_NODE_PROPERTIES.includes(name as any)) {
        throw new Error(
            `Volten Error: Output name "${name}" is a reserved property name.`
        );
    }
}

/**
 * Base internal properties for all Nodes.
 * These are always prefixed with "_" to avoid collision with output names.
 */
export interface NodeBase {
    /** Internal identifier */
    readonly _id: symbol;
    /** Human-friendly debug label */
    readonly _label: string;
    /** Dependencies (parent nodes providing inputs to this pass) */
    readonly _dependencies: readonly Node[];
    /** Reference to the kernel this node executes */
    readonly _kernel: Kernel;
    /** The cached compute pipeline */
    readonly _pipeline: GPUComputePipeline;
    /** The bind group layout from the pipeline */
    readonly _bindGroupLayout: GPUBindGroupLayout;
    /** Classified binding entries (for v.run to create bind groups) */
    readonly _bindingEntries: readonly BindingEntry[];
    /** Internal resources owned by this node and safe for Volten to destroy */
    readonly _ownedResources: readonly NodeOwnedResource[];
    /** Logical invocation bounds [x, y, z] before workgroup division */
    readonly _bounds: readonly [number, number, number];
    /** Dispatch dimensions [x, y, z] for workgroup dispatch */
    readonly _dispatch: readonly [number, number, number];
    /** Original user-provided bindings record (for Handle resolution at run time) */
    readonly _bindings: Readonly<Record<string, unknown>>;
    /** Full assembled shader code (for debugging) */
    readonly _shaderCode: string;
    /** Cached bind group for repeated executions of this node. */
    _cachedBindGroup?: CachedNodeBindGroup | null;
    /** Optional internal shader-debug state for this node */
    readonly _debug: NodeDebugState | null;
}

/**
 * Node in the compute DAG.
 * Represents a compute pass with its dependencies and output handles.
 *
 * Handles are spread directly onto the node for ALL buffer-like bindings
 * (Buffer, RawBuffer, Handle)
 *
 * @example
 * ```ts
 * // Declared outputs
 * const A = v.pass(BlurKernel, { in: source, out: dest });
 * const B = v.pass(SharpenKernel, { in: A.out }); // A.out is a Handle
 *
 * // In-place: no outputs declared, but handles still available
 * const C = v.pass(InPlaceKernel, { data: buf });
 * const D = v.pass(InPlaceKernel, { data: C.data }); // chaining works
 * ```
 *
 * @typeParam TOutputs - Record of output names to Handle types
 */
export type Node<
    TOutputs extends Record<string, Handle> = Record<string, Handle>
> = NodeBase & TOutputs;

/**
 * Helper to extract all public buffer handles from a Node (filtering out
 * internal properties).
 */
export function getNodeHandles(node: Node): Record<string, Handle> {
    const handles: Record<string, Handle> = {};
    for (const key of Object.keys(node)) {
        if (!key.startsWith('_')) {
            handles[key] = (node as any)[key];
        }
    }
    return handles;
}

/**
 * Helper to extract only the handles declared as kernel outputs.
 */
export function getNodeOutputHandles(node: Node): Record<string, Handle> {
    const handles = getNodeHandles(node);
    const outputs: Record<string, Handle> = {};

    for (const name of node._kernel.outputNames) {
        const handle = handles[name];
        if (!handle) {
            throw new Error(
                `Volten Error: Kernel output "${name}" was declared, but no matching buffer binding exists on node "${node._label}".\n` +
                    `  Hint: Provide a buffer binding named "${name}", or read an explicit handle with v.read(node.someHandle).`
            );
        }
        outputs[name] = handle;
    }

    return outputs;
}

/**
 * Creates a Handle for a node output.
 */
export function createHandle(node: Node, name: string): Handle {
    validateOutputName(name);
    return {
        _id: Symbol(`Handle:${name}`),
        _node: node,
        _name: name,
        _label: `${node._label}.${name}`
    };
}

/**
 * Options for creating a Node.
 */
export interface CreateNodeOptions {
    kernel: Kernel;
    pipeline: GPUComputePipeline;
    bindGroupLayout: GPUBindGroupLayout;
    bindingEntries: BindingEntry[];
    ownedResources?: NodeOwnedResource[];
    bounds: [number, number, number];
    dispatch: [number, number, number];
    bindings: Record<string, unknown>;
    shaderCode: string;
    dependencies: Node[];
    label?: string;
    debug?: NodeDebugState | null;
}

/**
 * Create a Node with handles spread directly onto it.
 *
 * This is the primary factory for Node objects. It:
 * 1. Creates the NodeBase with all internal properties
 * 2. Creates a Handle for each buffer-like binding (Buffer, RawBuffer, Handle)
 * 3. Spreads handles directly onto the node
 *
 * @param options - Node creation options
 * @returns A fully constructed Node with handles for all buffer-like bindings
 */
export function createNode(options: CreateNodeOptions): Node {
    const {
        kernel,
        pipeline,
        bindGroupLayout,
        bindingEntries,
        ownedResources = [],
        bounds,
        dispatch,
        bindings,
        shaderCode,
        dependencies,
        label,
        debug = null
    } = options;

    // Create the base node object (mutated to add handles below)
    const node: any = {
        _id: Symbol('Node'),
        _label: label ?? 'Node',
        _dependencies: Object.freeze(dependencies),
        _kernel: kernel,
        _pipeline: pipeline,
        _bindGroupLayout: bindGroupLayout,
        _bindingEntries: Object.freeze(bindingEntries),
        _ownedResources: Object.freeze([...ownedResources]),
        _bounds: Object.freeze(bounds) as readonly [number, number, number],
        _dispatch: Object.freeze(dispatch) as readonly [number, number, number],
        _bindings: Object.freeze(bindings),
        _shaderCode: shaderCode,
        _cachedBindGroup: null,
        _debug: debug
    };

    // Create and spread handles for ALL buffer-like bindings
    for (const [name, value] of Object.entries(bindings)) {
        if (isBufferLike(value)) {
            const handle = createHandle(node as Node, name);
            node[name] = handle;
        }
    }

    return node as Node;
}

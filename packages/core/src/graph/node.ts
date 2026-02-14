// Node and Handle types
// Opaque handles for DAG nodes and their outputs

import type { Kernel } from '../kernel/kernel.js';
import type { BindingEntry } from '../kernel/bindings.js';

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
        typeof (value as Handle)._id === 'symbol'
    );
}

// -----------------------------------------------------------------------------
// Node Design Philosophy:
// -----------------------------------------------------------------------------
// Output handles are spread directly onto the Node for API terseness:
//
//   const A = v.pass(K1, { in: input });
//   const B = v.pass(K2, { in: A.output });  // ✅ Direct access
//
// Instead of the more verbose:
//
//   const B = v.pass(K2, { in: A.outputs.output });  // ❌ Extra indirection
//
// Convention:
// - Internal properties are prefixed with "_" (e.g., _id, _dependencies)
// - User-facing output names must NOT start with "_" (validated at runtime)
// - This separation allows iteration over outputs via Object.keys filtering
// -----------------------------------------------------------------------------

/** Reserved property names that cannot be used as output names */
export const RESERVED_NODE_PROPERTIES = [
    '_id', '_dependencies', '_kernel',
    '_pipeline', '_bindGroupLayout', '_bindingEntries',
    '_dispatch', '_bindings', '_shaderCode',
] as const;

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
    /** Dispatch dimensions [x, y, z] for workgroup dispatch */
    readonly _dispatch: readonly [number, number, number];
    /** Original user-provided bindings record (for Handle resolution at run time) */
    readonly _bindings: Readonly<Record<string, unknown>>;
    /** Full assembled shader code (for debugging) */
    readonly _shaderCode: string;
}

/**
 * Node in the compute DAG.
 * Represents a compute pass with its dependencies and output handles.
 * 
 * Output handles are spread directly onto the node for terse access:
 * @example
 * ```ts
 * const A = v.pass(BlurKernel, { in: source });
 * const B = v.pass(SharpenKernel, { in: A.result }); // A.result is a Handle
 * ```
 * 
 * @typeParam TOutputs - Record of output names to Handle types
 */
export type Node<TOutputs extends Record<string, Handle> = Record<string, Handle>> =
    NodeBase & TOutputs;

/**
 * Helper to extract output handles from a Node (filtering out internal properties).
 */
export function getNodeOutputs(node: Node): Record<string, Handle> {
    const outputs: Record<string, Handle> = {};
    for (const key of Object.keys(node)) {
        if (!key.startsWith('_')) {
            outputs[key] = (node as any)[key];
        }
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
    dispatch: [number, number, number];
    bindings: Record<string, unknown>;
    shaderCode: string;
    dependencies: Node[];
}

/**
 * Create a Node with output handles spread directly onto it.
 * 
 * This is the primary factory for Node objects. It:
 * 1. Creates the NodeBase with all internal properties
 * 2. Creates a Handle for each kernel output
 * 3. Spreads output handles directly onto the node
 * 
 * @param options - Node creation options
 * @returns A fully constructed Node with output handles
 */
export function createNode(options: CreateNodeOptions): Node {
    const {
        kernel, pipeline, bindGroupLayout, bindingEntries,
        dispatch, bindings, shaderCode, dependencies,
    } = options;

    // Create the base node object (mutated to add handles below)
    const node: any = {
        _id: Symbol('Node'),
        _dependencies: Object.freeze(dependencies),
        _kernel: kernel,
        _pipeline: pipeline,
        _bindGroupLayout: bindGroupLayout,
        _bindingEntries: Object.freeze(bindingEntries),
        _dispatch: Object.freeze(dispatch) as readonly [number, number, number],
        _bindings: Object.freeze(bindings),
        _shaderCode: shaderCode,
    };

    // Create and spread output handles onto the node
    for (const output of kernel.outputs) {
        const handle = createHandle(node as Node, output.name);
        node[output.name] = handle;
    }

    return node as Node;
}

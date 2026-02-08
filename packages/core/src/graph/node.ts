// Node and Handle types
// Opaque handles for DAG nodes and their outputs

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
export const RESERVED_NODE_PROPERTIES = ['_id', '_dependencies', '_kernel'] as const;

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
    readonly _kernel: unknown; // Will be typed as Kernel once that module exists
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

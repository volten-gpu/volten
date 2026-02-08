// Node and Handle types
// Opaque handles for DAG nodes and their outputs

/**
 * Handle to a buffer output from a compute pass
 * Opaque type - users shouldn't rely on internal structure
 */
export interface Handle {
    /** Internal identifier */
    readonly _id: symbol;
    /** Reference to parent node */
    readonly _node: Node;
}

/**
 * Node in the compute DAG
 * Represents a compute pass with its dependencies
 */
export interface Node {
    /** Internal identifier */
    readonly _id: symbol;
    /** Output handles keyed by name */
    readonly outputs: Record<string, Handle>;
    /** Dependencies (parent nodes) */
    readonly _dependencies: Node[];
}

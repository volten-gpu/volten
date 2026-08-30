// DAG Scheduler — Kahn's algorithm with priority-based tiebreaking
// Given one or more root nodes, walks the dependency graph backwards,
// then produces a safe execution order via Kahn's algorithm.
// When multiple nodes become free simultaneously, an optional priority
// map is used as tiebreaker (lower priority number → earlier execution).

import type { DispatchNode } from './dispatch-node.js';

/**
 * Collect all reachable nodes by walking dependencies from a terminal node.
 * Uses iterative DFS to avoid stack overflows on deep graphs.
 */
export function collectNodes(terminal: DispatchNode): DispatchNode[] {
    const visited = new Set<symbol>();
    const nodes: DispatchNode[] = [];
    const stack: DispatchNode[] = [terminal];

    while (stack.length > 0) {
        const node = stack.pop()!;
        if (visited.has(node._id)) continue;
        visited.add(node._id);
        nodes.push(node);

        for (const dep of node._dependencies) {
            if (!visited.has(dep._id)) {
                stack.push(dep);
            }
        }
    }

    return nodes;
}

/**
 * Collect all reachable nodes from multiple roots, deduplicating.
 */
export function collectNodesFromMultiple(
    roots: readonly DispatchNode[]
): DispatchNode[] {
    const visited = new Set<symbol>();
    const nodes: DispatchNode[] = [];
    const stack: DispatchNode[] = [...roots];

    while (stack.length > 0) {
        const node = stack.pop()!;
        if (visited.has(node._id)) continue;
        visited.add(node._id);
        nodes.push(node);

        for (const dep of node._dependencies) {
            if (!visited.has(dep._id)) {
                stack.push(dep);
            }
        }
    }

    return nodes;
}

/**
 * Topologically sort the DAG reachable from one or more root nodes
 * using Kahn's algorithm.
 *
 * Returns nodes in execution order: sources first, terminals last.
 * Throws if a cycle is detected.
 *
 * When multiple nodes become free at the same time (in-degree drops to 0
 * simultaneously), the optional `priority` map determines the order.
 * Lower priority values are executed first. This is used by the compiler
 * to enforce positional order from v.run(A, B, C): nodes affiliated with
 * earlier terminals get lower priority values and are scheduled first.
 *
 * @param roots - One or more entry-point nodes (terminals whose subtrees should be sorted)
 * @param priority - Optional map from node._id to priority number (lower = earlier).
 *                   When omitted, ties are broken by insertion order (FIFO).
 * @returns Nodes in topological (execution) order
 */
export function topologicalSort(
    roots: readonly DispatchNode[],
    priority?: Map<symbol, number>
): DispatchNode[] {
    const allNodes = collectNodesFromMultiple(roots);

    // Build in-degree map (only counting edges within the reachable subgraph)
    const reachable = new Set(allNodes.map((n) => n._id));
    const inDegree = new Map<symbol, number>();

    // Count in-degrees: for each node, count how many of its dependencies are in the subgraph
    for (const node of allNodes) {
        let count = 0;
        for (const dep of node._dependencies) {
            if (reachable.has(dep._id)) {
                count++;
            }
        }
        inDegree.set(node._id, count);
    }

    // Index by id for fast lookup
    const nodeById = new Map<symbol, DispatchNode>();
    for (const node of allNodes) {
        nodeById.set(node._id, node);
    }

    // Build forward adjacency: dep._id → list of node._id that depend on it
    const dependents = new Map<symbol, symbol[]>();
    for (const node of allNodes) {
        for (const dep of node._dependencies) {
            if (reachable.has(dep._id)) {
                let list = dependents.get(dep._id);
                if (!list) {
                    list = [];
                    dependents.set(dep._id, list);
                }
                list.push(node._id);
            }
        }
    }

    // Helper: sort an array of nodes by priority (lower first), stable for equal priorities
    // why is this necessary? imagine this example:
    // A = k({ inout: buffer1 });
    // B = k({ inout: buffer1 });
    // C = k({ inout: A.inout });
    // v.run(A, B, C);
    // ^ without sortByPriority, A and B will simultaneously become free,
    // and will both be in-degree-0 at the same time, however without sorting by
    // priority, B will be scheduled before A, but the user intent was to
    // schedule A before B. To fix this when nodes become free we're sorting them
    // such that they're ordered in the same way in which they appear when
    // calling v.run(...)
    const sortByPriority = (nodes: DispatchNode[]): void => {
        if (priority && nodes.length > 1) {
            nodes.sort(
                (a, b) =>
                    (priority.get(a._id) ?? Infinity) -
                    (priority.get(b._id) ?? Infinity)
            );
        }
    };

    // Kahn's algorithm: start with all nodes that have in-degree 0, sorted by priority
    const queue: DispatchNode[] = allNodes.filter(
        (n) => inDegree.get(n._id) === 0
    );
    sortByPriority(queue);

    const sorted: DispatchNode[] = [];
    while (queue.length > 0) {
        const node = queue.shift()!;
        sorted.push(node);

        const deps = dependents.get(node._id);
        if (deps) {
            const newlyFreed: DispatchNode[] = [];
            for (const depId of deps) {
                const degree = inDegree.get(depId)! - 1;
                inDegree.set(depId, degree);
                if (degree === 0) {
                    newlyFreed.push(nodeById.get(depId)!);
                }
            }
            // Tiebreak newly freed nodes by priority before adding to queue
            sortByPriority(newlyFreed);
            queue.push(...newlyFreed);
        }
    }

    if (sorted.length !== allNodes.length) {
        throw new Error(
            'Volten Error: Cycle detected in compute graph. ' +
                `Expected ${allNodes.length} nodes but only ${sorted.length} could be sorted. ` +
                'Check that no node depends on itself (directly or indirectly).'
        );
    }

    return sorted;
}

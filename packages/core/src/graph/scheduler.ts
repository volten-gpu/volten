// DAG Scheduler — Kahn's algorithm for topological sort
// Given a terminal node, walks the dependency graph backwards,
// then produces a safe execution order via Kahn's algorithm.

import type { Node } from './node.js';

/**
 * Collect all reachable nodes by walking dependencies from a terminal node.
 * Uses iterative DFS to avoid stack overflows on deep graphs.
 */
function collectNodes(terminal: Node): Node[] {
    const visited = new Set<symbol>();
    const nodes: Node[] = [];
    const stack: Node[] = [terminal];

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
 * Topologically sort the DAG reachable from a terminal node using Kahn's algorithm.
 *
 * Returns nodes in execution order: sources first, terminal last.
 * Throws if a cycle is detected.
 *
 * @param terminal - The final node whose result is needed
 * @returns Nodes in topological (execution) order
 */
export function topologicalSort(terminal: Node): Node[] {
    const allNodes = collectNodes(terminal);

    // Build in-degree map (only counting edges within the reachable subgraph)
    const reachable = new Set(allNodes.map(n => n._id));
    const inDegree = new Map<symbol, number>();
    for (const node of allNodes) {
        if (!inDegree.has(node._id)) {
            inDegree.set(node._id, 0);
        }
        for (const dep of node._dependencies) {
            if (reachable.has(dep._id)) {
                // dep → node edge exists, but we count in-degree for node
                // (i.e. node depends on dep, so node has an incoming edge from dep)
            }
        }
    }

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
    const nodeById = new Map<symbol, Node>();
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

    // Kahn's algorithm: start with all nodes that have in-degree 0
    const queue: Node[] = [];
    for (const node of allNodes) {
        if (inDegree.get(node._id) === 0) {
            queue.push(node);
        }
    }

    const sorted: Node[] = [];
    while (queue.length > 0) {
        const node = queue.shift()!;
        sorted.push(node);

        const deps = dependents.get(node._id);
        if (deps) {
            for (const depId of deps) {
                const degree = inDegree.get(depId)! - 1;
                inDegree.set(depId, degree);
                if (degree === 0) {
                    queue.push(nodeById.get(depId)!);
                }
            }
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

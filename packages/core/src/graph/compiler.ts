// ============================================================================
// Graph Compiler — Multi-Terminal Merge & Synthetic Dependency Injection
// ============================================================================
//
// This module sits between graph construction (v.pass) and GPU execution
// (_submit). Its job is to take one or more terminal nodes and produce
// a single, correctly-ordered ExecutionPlan.
//
// # Why this exists
//
// When the user writes:
//
//   v.run(D, E, F)
//
// each terminal (by terminal node we refer to D, E, F in the example) might be the tip of its own subtree.
// Some subtrees may share concrete buffers without any Handle-based connection:
//
//   let E = v.pass(k2, { inout });          // uses buffer "inout"
//   let K = v.pass(k6, { in: inout });      // also uses buffer "inout"
//   let L = v.pass(k6, { in: K.in });
//   v.run(E, L);
//
// Here, E and K both touch "inout" but have NO Handle-based dependency.
// Without intervention, topological sort could execute K before E,
// producing incorrect results (K would read stale data).
// Another problem is that E and K could be executed within the same compute pass,
// causing Read-Write conflicts.
//
// The compiler detects shared concrete buffers between independent
// subtrees and injects "synthetic dependencies" to enforce the
// positional order specified in v.run().
//
// # When synthetic deps are NOT needed
//
// If two terminals share a subtree through Handle-based connections
// (e.g., v.run(D,G) and both D and G depend on C via C.out), the dependency is
// already captured. No synthetic dep is injected — the topological sort
// naturally places C before both D and G, and D/G remain unordered
// relative to each other (they can run in parallel).
//
// # Priority-based tiebreaking
//
// Even after synthetic deps are injected, Kahn's algorithm may encounter
// multiple nodes that are simultaneously free. The compiler assigns each
// node a priority equal to the minimum terminal index it belongs to.
// This priority is passed to topologicalSort as a tiebreaker, ensuring
// that v.run(A, B, C) produces execution order consistent with the user's
// positional intent at every level of the graph.
//
// # Volten auto-allocated buffers
//
// Volten auto-allocated buffers (future feature) are exempt from overlap
// detection because they only exist through Handle references in the
// graph. Two independent nodes can never accidentally share a
// Volten-owned buffer — Volten controls the allocation and would
// never assign the same buffer to two live nodes.
//
// ============================================================================

import type { Node } from './node.js';
import { collectNodes, topologicalSort } from './scheduler.js';
import { resolveConcreteBuffer } from '../kernel/resource-resolution.js';
import type { Buffer } from '../data/buffer.js';
import type { RawBuffer } from '../data/raw-buffer.js';
export { resolveConcreteBuffer } from '../kernel/resource-resolution.js';

/**
 * The output of the compile step. Contains a sorted list of nodes
 * ready for GPU encoding.
 */
export interface ExecutionPlan {
    /** Nodes in topological (execution) order. */
    readonly sorted: readonly Node[];
}

/**
 * Compile one or more terminal nodes into an ExecutionPlan.
 *
 * This is the central scheduling function. It:
 *
 * 1. Iterates terminals in positional order
 * 2. Collects each terminal's subtree
 * 3. **Detects buffer overlap** — for each subtree, checks if any node uses
 *    a concrete buffer that was already used by a node exclusive to a
 *    previous terminal's subtree. Shared nodes (appearing in multiple
 *    subtrees) are excluded from overlap checks since they're already
 *    connected via Handle-based dependencies.
 * 4. **Injects synthetic dependencies** — when overlap is found between
 *    exclusive nodes of different terminals, the previous terminal becomes
 *    a dependency of the current terminal. This ensures Kahn's algorithm
 *    places the previous terminal's subtree first.
 * 5. **Builds a priority map** — each node gets a priority equal to the
 *    minimum terminal index it belongs to. This is passed to topologicalSort
 *    as a tiebreaker so that nodes affiliated with earlier terminals are
 *    scheduled first.
 * 6. **Topologically sorts** — the (possibly augmented) terminal list is
 *    sorted directly
 *
 * @param terminals - Terminal nodes in the order specified by v.run()
 * @returns An ExecutionPlan with nodes in safe execution order
 *
 * @example
 * ```ts
 * // Independent trees with shared buffer:
 * let E = v.pass(k2, { inout });
 * let K = v.pass(k6, { in: inout });
 * let L = v.pass(k6, { in: K.in });
 * compile([E, L]) // → [E, K, L] (E before K due to shared "inout")
 * ```
 */
export function compile(terminals: Node[]): ExecutionPlan {
    // Question: why is this function necessary? can't we simply use the topological sort with the
    // graph-api?
    // Answer: No, because it woulnd't track this kind of dependencies:
    // A = v.pass(k, { inout: buffer1 });
    // B = v.pass(k, { inout: buffer1 });
    // C = v.pass(k, { inout: buffer1 });
    // v.run(A, B, C);
    //
    // ^ these passes can't run all in parallel while keeping the same Pass open and adding multiple
    // dispatches, because we would risk Read-Write conflicts, this is the reason why this function exists,
    // it tries to find the implicit dependencies based on which buffers are being used by which node.
    // when it finds these dependencies, it will inject synthetic graph-dependencies and only then we run
    // the topological sort

    if (terminals.length === 0) {
        throw new Error(
            'Volten Error: compile() called with no terminal nodes.'
        );
    }

    // Fast path: single terminal, no merging needed
    if (terminals.length === 1) {
        return { sorted: topologicalSort([terminals[0]]) };
    }

    // -----------------------------------------------------------------------
    // Phase 1: Discover subtrees + identify shared vs exclusive nodes
    // -----------------------------------------------------------------------

    // Each node X in this map contains a set of all the terminal indices where X appears in their subtree
    /*
            A              G                   F           <-- terminal nodes (A,G,F)
           / \            /                     \
          B   C          C  <-- set = (G,A)      K  <-- set = (F)
         / \                                      \
        D   E  <-- set(A)                          D  <-- set = (A, F) 
    */
    const nodeOwnership = new Map<symbol, Set<number>>();
    const subtrees: Node[][] = [];

    for (let i = 0; i < terminals.length; i++) {
        const subtree = collectNodes(terminals[i]);
        subtrees.push(subtree);
        for (const node of subtree) {
            let owners = nodeOwnership.get(node._id);
            if (!owners) {
                owners = new Set();
                nodeOwnership.set(node._id, owners);
            }
            owners.add(i);
        }
    }

    // A node is "exclusive" to a terminal if it only appears in that
    // terminal's subtree. Shared nodes are already connected via Handles
    // and don't need synthetic deps.
    /*
            A           G
           / \         /
          B   C       C   <--- C is not exclusive to A or G since it appears in both subtrees
         / \   
        D   E    <-- E is exclusive to A since E only appears in the subtree of A
    */
    const isExclusive = (nodeId: symbol, terminalIndex: number): boolean => {
        const owners = nodeOwnership.get(nodeId);
        return (
            owners !== undefined &&
            owners.size === 1 &&
            owners.has(terminalIndex)
        );
    };

    // -----------------------------------------------------------------------
    // Phase 2: Detect buffer overlap between exclusive nodes
    // -----------------------------------------------------------------------

    // Maps concrete buffer → the terminal index that "owns" it (most recent
    // terminal whose exclusive nodes use this buffer).
    // When we find a conflict, we inject a synthetic dep.
    const bufferToTerminal = new Map<Buffer | RawBuffer, number>();

    // Track which terminals need synthetic deps: syntheticDeps[i] contains
    // terminal indices that must complete before terminal i.
    const syntheticDeps: Set<number>[] = terminals.map(() => new Set());

    for (let i = 0; i < terminals.length; i++) {
        for (const node of subtrees[i]) {
            // Only check exclusive nodes — shared nodes are already
            // connected via Handle-based dependencies
            if (!isExclusive(node._id, i)) continue;

            for (const value of Object.values(node._bindings)) {
                const concrete = resolveConcreteBuffer(value);
                if (concrete === null) continue;

                const previousOwner = bufferToTerminal.get(concrete);
                if (previousOwner !== undefined && previousOwner !== i) {
                    // Conflict: this buffer was used by exclusive nodes
                    // of a previous terminal. Inject synthetic dep.
                    syntheticDeps[i].add(previousOwner);
                }
                // Claim this buffer for the current terminal
                bufferToTerminal.set(concrete, i);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase 3: Inject synthetic dependencies into terminal nodes
    // -----------------------------------------------------------------------

    const finalTerminals: Node[] = [];

    for (let i = 0; i < terminals.length; i++) {
        if (syntheticDeps[i].size > 0) {
            // This terminal needs synthetic deps — create a wrapper node
            // that has the original deps + the synthetic ones.
            // Use finalTerminals (not originals) so chained synthetic deps
            // compose correctly: if G depends on F and F depends on E,
            // G's dep must point to the wrapped F that already includes E.
            const extraDeps = [...syntheticDeps[i]].map(
                (j) => finalTerminals[j]
            );
            const original = terminals[i];
            const wrappedDeps = [...original._dependencies, ...extraDeps];

            // Shallow clone with augmented dependencies
            const wrapped: Node = Object.create(null);
            for (const key of Object.keys(original) as (keyof Node)[]) {
                (wrapped as any)[key] = (original as any)[key];
            }
            // Also copy symbol-keyed and prototype properties won't matter
            // since we only need _id, _dependencies, etc.
            (wrapped as any)._dependencies = Object.freeze(wrappedDeps);
            finalTerminals.push(wrapped);
        } else {
            finalTerminals.push(terminals[i]);
        }
    }

    // -----------------------------------------------------------------------
    // Phase 4: Build priority map + topological sort
    // -----------------------------------------------------------------------

    // Each node's priority is the minimum terminal index it belongs to.
    // This ensures that when multiple nodes become free simultaneously,
    // nodes affiliated with earlier terminals are scheduled first.
    const priority = new Map<symbol, number>();
    for (const [nodeId, owners] of nodeOwnership) {
        priority.set(nodeId, Math.min(...owners));
    }

    const sorted = topologicalSort(finalTerminals, priority);

    return { sorted };
}

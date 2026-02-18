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
// # Pool-allocated buffers
//
// Pool-allocated buffers (future feature) are exempt from overlap
// detection because they only exist through Handle references in the
// graph. Two independent nodes can never accidentally share a
// pool-allocated buffer — Volten controls the allocation and would
// never assign the same buffer to two live nodes.
//
// ============================================================================

import type { Node } from './node.js';
import { collectNodes } from './scheduler.js';
import { topologicalSort } from './scheduler.js';
import { Buffer } from '../data/buffer.js';
import { RawBuffer } from '../data/raw-buffer.js';
import { isHandle, type Handle } from './node.js';

/**
 * The output of the compile step. Contains a sorted list of nodes
 * ready for GPU encoding.
 */
export interface ExecutionPlan {
    /** Nodes in topological (execution) order. */
    readonly sorted: readonly Node[];
}

/**
 * Walk a Handle chain to find the root Buffer or RawBuffer.
 *
 * Returns null if the binding is not buffer-like (e.g., a scalar uniform
 * that was somehow passed through, or a future PoolSlot placeholder).
 *
 * @example
 * ```
 * // A = v.pass(k, { in: buf, out: outBuf })
 * // B = v.pass(k, { in: A.out })
 *
 * resolveConcreteBuffer(buf)    // → buf (Buffer)
 * resolveConcreteBuffer(A.out)  // → outBuf (Buffer) — walks A._bindings['out']
 * resolveConcreteBuffer(B.in)   // → outBuf (Buffer) — walks B._bindings['in'] → A.out → outBuf
 * ```
 */
export function resolveConcreteBuffer(value: unknown): Buffer | RawBuffer | null {
    if (value instanceof Buffer || value instanceof RawBuffer) {
        return value;
    }
    if (isHandle(value)) {
        const parentBinding = value._node._bindings[value._name];
        return resolveConcreteBuffer(parentBinding);
    }
    return null;
}

/**
 * Collect all unique concrete buffers used by a set of nodes.
 * Walks each node's bindings and resolves Handles to their root buffers.
 *
 * @param nodes - Nodes to inspect (typically all nodes in a subtree)
 * @returns Set of concrete Buffer/RawBuffer instances used by these nodes
 */
function collectConcreteBuffers(nodes: Node[]): Set<Buffer | RawBuffer> {
    const buffers = new Set<Buffer | RawBuffer>();
    for (const node of nodes) {
        for (const value of Object.values(node._bindings)) {
            const concrete = resolveConcreteBuffer(value);
            if (concrete !== null) {
                buffers.add(concrete);
            }
        }
    }
    return buffers;
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
 * 5. **Creates a virtual sentinel** — an ephemeral node that depends on
 *    all terminals (including those with injected synthetic deps). This
 *    gives topologicalSort a single entry point for the merged graph.
 * 6. **Strips the sentinel** — the sentinel is removed from the sorted
 *    output since it's not a real compute node.
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
    if (terminals.length === 0) {
        throw new Error(
            'Volten Error: compile() called with no terminal nodes.'
        );
    }

    // Fast path: single terminal, no merging needed
    if (terminals.length === 1) {
        return { sorted: topologicalSort(terminals[0]) };
    }

    // -----------------------------------------------------------------------
    // Phase 1: Discover subtrees + identify shared vs exclusive nodes
    // -----------------------------------------------------------------------

    // Each node X in this map contains a set of all the terminal indices where X appears in their subtree
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
        return owners !== undefined && owners.size === 1 && owners.has(terminalIndex);
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
    // Phase 3: Create virtual sentinel with merged dependencies
    // -----------------------------------------------------------------------

    // The sentinel depends on all terminals. Additionally, for terminals
    // with synthetic deps, we inject the dependency directly on the
    // terminal node so topologicalSort sees the edge.
    //
    // We need to clone terminals that gain synthetic deps because Node
    // has frozen dependencies. We create thin wrappers that add the
    // extra deps without mutating the originals.

    const finalTerminals: Node[] = [];

    for (let i = 0; i < terminals.length; i++) {
        if (syntheticDeps[i].size > 0) {
            // This terminal needs synthetic deps — create a wrapper node
            // that has the original deps + the synthetic ones
            // Use finalTerminals (not originals) so chained synthetic deps
            // compose correctly: if G depends on F and F depends on E,
            // G's dep must point to the wrapped F that already includes E.
            const extraDeps = [...syntheticDeps[i]].map(j => finalTerminals[j]);
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

    // Virtual sentinel: depends on all (possibly wrapped) terminals.
    // This gives topologicalSort a single entry point.
    const sentinel: Node = {
        _id: Symbol('Sentinel'),
        _dependencies: Object.freeze(finalTerminals),
        _kernel: {} as any,
        _pipeline: {} as any,
        _bindGroupLayout: {} as any,
        _bindingEntries: Object.freeze([]),
        _dispatch: Object.freeze([1, 1, 1]) as readonly [number, number, number],
        _bindings: Object.freeze({}),
        _shaderCode: '',
    } as any as Node;

    // -----------------------------------------------------------------------
    // Phase 4: Topological sort the merged graph
    // -----------------------------------------------------------------------

    const sorted = topologicalSort(sentinel);

    // Strip the sentinel — it's not a real compute node
    const result = sorted.filter(n => n._id !== sentinel._id);

    return { sorted: result };
}

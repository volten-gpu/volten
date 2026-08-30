/**
 * Turns materialized invocations into one ordered list of GPU dispatches.
 *
 * # Invocations and terminals
 *
 * Each item passed to v.run() becomes one invocation:
 *
 *     v.run([A, B]);
 *            |  |
 *            0  1
 *
 * A kernel invocation has one terminal. A plan may have more than one:
 *
 *   Invocation A                 Invocation B
 *
 *   R1 -> R2 -> R3               B1 -> B2
 *   H1 ------> H2
 *
 *   terminals: [R3, H2]          terminals: [B2]
 *
 * R3 and H2 are both ends of Invocation A. The array does not add a
 * dependency between them. They stay independent unless their handles
 * already connect them.
 *
 * If two terminals inside one plan need an order, the plan must use a handle:
 *
 *   const A = update({ data });
 *   const B = update({ data: A.data });
 *
 *   terminals: [A, B]     A -> B
 *
 * The order of properties returned by a plan is not an execution order.
 *
 * # Why topologicalSort() is not enough
 *
 * Handles create normal graph dependencies:
 *
 *   const A = update({ data });
 *   const B = update({ data: A.data });
 *
 *   A -> B
 *
 * Sometimes two invocations use the same Buffer directly:
 *
 *   const A = update({ data });
 *   const B = update({ data });
 *   v.run([A, B]);
 *
 *   A(data)       B(data)
 *      no handle edge
 *
 * A and B still need the order given to v.run(). A topological sort cannot
 * discover that order because the graph has no edge between them.
 *
 * This compiler fills that gap. It:
 *
 * 1. Finds every dispatch used by each invocation.
 * 2. Finds concrete buffers used by separate invocations.
 * 3. Adds temporary dependencies where v.run() order must be kept.
 * 4. Topologically sorts all terminals into one dispatch list.
 *
 * "Temporary" means these dependencies only exist in this compile call.
 * The cached dispatch graph is not changed.
 *
 * # Future allocated buffers
 *
 * A future Volten-owned buffer should normally move through handles. Those
 * handles already describe its order. This shared-buffer check is mainly for
 * concrete buffers passed directly to separate invocations.
 */

import type { DispatchNode } from './dispatch-node.js';
import { collectNodesFromMultiple, topologicalSort } from './scheduler.js';
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
    readonly sorted: readonly DispatchNode[];
}

/**
 * The physical ends of one logical node passed to v.run().
 *
 * Multiple terminals belong to the same invocation, but they do not depend
 * on each other just because they are in this array.
 */
export interface MaterializedInvocation {
    readonly terminals: readonly DispatchNode[];
}

interface InvocationCompileState {
    readonly invocation: MaterializedInvocation;
    readonly reachableNodes: readonly DispatchNode[];
    /** Earlier invocations that must finish first. */
    readonly requiredInvocations: Set<number>;
}

/**
 * Merges materialized invocations and returns a safe dispatch order.
 */
export function compile(
    invocations: readonly MaterializedInvocation[]
): ExecutionPlan {
    if (invocations.length === 0) {
        throw new Error(
            'Volten Error: compile() called with no materialized invocations.'
        );
    }

    // With one invocation, handle dependencies already describe its order.
    if (invocations.length === 1) {
        return { sorted: topologicalSort(invocations[0].terminals) };
    }

    // Phase 1: find every dispatch used by each invocation.
    //
    // A dispatch may be reachable from more than one invocation:
    //
    //   Invocation 0                         Invocation 1
    //   terminals: [D]                      terminals: [G]
    //
    //                A -> B -> C
    //                          | \
    //                          D  G
    //
    // Walking back from D finds A, B, C and D.
    // Walking back from G finds A, B, C and G.
    //
    // Ownership:
    //
    //   A, B, C -> { 0, 1 }   shared
    //   D       -> { 0 }      exclusive to invocation 0
    //   G       -> { 1 }      exclusive to invocation 1
    //
    // A larger example can share different nodes:
    //
    //   Invocation 0       Invocation 1       Invocation 2
    //   terminals: [A]     terminals: [G]     terminals: [F]
    //
    //   A                  G                  F
    //   +-- B              +-- C              +-- K
    //   |   +-- D                                 +-- D
    //   |   +-- E
    //   +-- C
    //
    // The two C labels are the same dispatch.
    // The two D labels are also the same dispatch.
    //
    //   C -> { 0, 1 }
    //   D -> { 0, 2 }
    //   E -> { 0 }
    //   K -> { 2 }
    //
    // A multi-terminal invocation uses the same rule:
    //
    //   Invocation 0
    //   terminals: [D, G]
    //
    //                A -> B -> C
    //                          | \
    //                          D  G
    //
    // collectNodesFromMultiple([D, G]) returns A, B, C, D and G once.
    // It does not add an edge between D and G.
    const nodeOwnership = new Map<symbol, Set<number>>();
    const states: InvocationCompileState[] = invocations.map((invocation) => ({
        invocation,
        reachableNodes: collectNodesFromMultiple(invocation.terminals),
        requiredInvocations: new Set()
    }));

    for (let i = 0; i < states.length; i++) {
        // foreach reachable-node from terminal i
        // (although terminal is not the perfect terminology, "invocation" is better)
        for (const node of states[i].reachableNodes) {
            let owners = nodeOwnership.get(node._id);
            if (!owners) {
                owners = new Set();
                nodeOwnership.set(node._id, owners);
            }
            owners.add(i);
        }
    }

    // Shared dispatches are already connected through handles. Checking them
    // again could create an order that the graph does not need. The buffer
    // check below only looks at exclusive dispatches.
    const isExclusive = (nodeId: symbol, invocationIndex: number): boolean => {
        const owners = nodeOwnership.get(nodeId);
        return (
            owners !== undefined &&
            owners.size === 1 &&
            owners.has(invocationIndex)
        );
    };

    // Phase 2: find separate invocations that use the same concrete buffer.
    //
    // Example:
    //
    //   Invocation 0             Invocation 1
    //   terminals: [E]           terminals: [L]
    //
    //   E(sharedBuffer)          K(sharedBuffer) -> L
    //
    // this is equivalent to:
    // let E = k({ input });
    // let K = k({ input });
    // let L = k({ K.input });
    //
    //
    // E and K have no handle edge. Their shared Buffer tells us that
    // Invocation 0 must stay before Invocation 1.

    // Remember the last invocation that used each concrete buffer.
    const bufferToInvocation = new Map<Buffer | RawBuffer, number>();

    for (let i = 0; i < states.length; i++) {
        const state = states[i];
        for (const node of state.reachableNodes) {
            if (!isExclusive(node._id, i)) continue;

            for (const value of Object.values(node._bindings)) {
                const concrete = resolveConcreteBuffer(value);
                if (concrete === null) continue;

                const previousOwner = bufferToInvocation.get(concrete);
                if (previousOwner !== undefined && previousOwner !== i) {
                    state.requiredInvocations.add(previousOwner);
                }
                bufferToInvocation.set(concrete, i);
            }
        }
    }

    const compiledInvocations: MaterializedInvocation[] = [];

    // Phase 3: add temporary dependencies to invocation terminals.
    //
    // If Invocation 0 must finish before Invocation 1:
    //
    //   Invocation 0               Invocation 1 before wrapping
    //   terminals: [R3, H2]        terminals: [B2, D4]
    //
    // Each terminal in Invocation 1 gets both earlier terminals:
    //
    //   R3 ---------> B2'
    //   H2 ---------> B2'
    //                  + original B2 dependencies
    //
    //   R3 ---------> D4'
    //   H2 ---------> D4'
    //                  + original D4 dependencies
    //
    // R3 and H2 still do not depend on each other. The new edges only keep
    // the two v.run() invocations in the requested order.
    for (const state of states) {
        if (state.requiredInvocations.size > 0) {
            // Use terminals that were already wrapped. This keeps longer
            // chains intact:
            //
            //   v.run([A, B, C]) with one shared buffer
            //
            //   A -> B' -> C'
            const extraDeps = [...state.requiredInvocations].flatMap(
                (j) => compiledInvocations[j].terminals
            );
            const wrappedTerminals = state.invocation.terminals.map(
                (original) => {
                    const wrappedDeps = [
                        ...original._dependencies,
                        ...extraDeps
                    ];

                    // Clone the terminal so another v.run() call can compile
                    // the same graph in a different order.
                    const wrapped: DispatchNode = Object.create(null);
                    for (const key of Object.keys(
                        original
                    ) as (keyof DispatchNode)[]) {
                        (wrapped as any)[key] = (original as any)[key];
                    }
                    (wrapped as any)._dependencies = Object.freeze(wrappedDeps);
                    return wrapped;
                }
            );
            compiledInvocations.push({ terminals: wrappedTerminals });
        } else {
            compiledInvocations.push(state.invocation);
        }
    }

    // Phase 4: sort the graph.
    //
    // Several dispatches may be ready at the same time. In that case, prefer
    // dispatches from earlier v.run() arguments. Dispatches in the same
    // invocation have the same priority, so this does not order sibling
    // terminals such as R3 and H2.
    const priority = new Map<symbol, number>();
    for (const [nodeId, owners] of nodeOwnership) {
        priority.set(nodeId, Math.min(...owners));
    }

    const sorted = topologicalSort(
        compiledInvocations.flatMap((invocation) => invocation.terminals),
        priority
    );

    return { sorted };
}

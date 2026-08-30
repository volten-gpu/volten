import { describe, it, expect } from 'vitest';
import {
    compile,
    resolveConcreteBuffer,
    type ExecutionPlan,
    type MaterializedInvocation
} from '../src/graph/compiler.js';
import {
    createDispatchNode as createNode,
    type DispatchNode as Node,
    type DispatchHandle as Handle
} from '../src/graph/dispatch-node.js';
import { Buffer } from '../src/data/buffer.js';
import { RawBuffer } from '../src/data/raw-buffer.js';

// ============================================================================
// Test helpers
// ============================================================================

/**
 * Create a minimal Node for testing.
 *
 * Uses real Buffer instances for bindings so that resolveConcreteBuffer
 * can walk Handle chains and detect buffer overlap correctly.
 */
function makeNode(
    idStr: string,
    bindings: Record<string, Buffer | RawBuffer | Handle> = {},
    dependencies: Node[] = []
): Node {
    return createNode({
        kernel: { _source: '', _options: {} } as any,
        pipeline: {} as any,
        bindGroupLayout: {} as any,
        bindingEntries: [],
        bounds: [1, 1, 1],
        dispatch: [1, 1, 1],
        bindings,
        shaderCode: '',
        dependencies
    });
}

/**
 * Assert that nodeA appears before nodeB in the sorted plan.
 */
function assertBefore(plan: ExecutionPlan, nodeA: Node, nodeB: Node): void {
    const idxA = plan.sorted.findIndex((n) => n._id === nodeA._id);
    const idxB = plan.sorted.findIndex((n) => n._id === nodeB._id);
    expect(idxA).toBeGreaterThanOrEqual(0);
    expect(idxB).toBeGreaterThanOrEqual(0);
    expect(idxA).toBeLessThan(idxB);
}

function invocation(...terminals: Node[]): MaterializedInvocation {
    return { terminals };
}

// ============================================================================
// resolveConcreteBuffer
// ============================================================================

describe('resolveConcreteBuffer', () => {
    it('returns a Buffer directly', () => {
        const buf = new Buffer([1, 2, 3], 'f32');
        expect(resolveConcreteBuffer(buf)).toBe(buf);
    });

    it('returns a RawBuffer directly', () => {
        const raw = new RawBuffer(
            new Float32Array([1, 2]).buffer,
            'array<f32>'
        );
        expect(resolveConcreteBuffer(raw)).toBe(raw);
    });

    it('walks a single Handle to its concrete Buffer', () => {
        const buf = new Buffer([1, 2, 3], 'f32');
        const node = makeNode('A', { data: buf });
        // node.data is a Handle pointing to buf
        expect(resolveConcreteBuffer(node.data)).toBe(buf);
    });

    it('walks a multi-level Handle chain', () => {
        const buf = new Buffer([1], 'f32');
        const A = makeNode('A', { out: buf });
        const B = makeNode('B', { input: A.out }, [A]);
        // B.input is a Handle → A.out Handle → buf
        expect(resolveConcreteBuffer(B.input)).toBe(buf);
    });

    it('walks a deep Handle chain (3 levels)', () => {
        const buf = new Buffer([1], 'f32');
        const A = makeNode('A', { out: buf });
        const B = makeNode('B', { data: A.out }, [A]);
        const C = makeNode('C', { src: B.data }, [B]);
        expect(resolveConcreteBuffer(C.src)).toBe(buf);
    });

    it('returns null for non-buffer values', () => {
        expect(resolveConcreteBuffer(42)).toBeNull();
        expect(resolveConcreteBuffer('hello')).toBeNull();
        expect(resolveConcreteBuffer(null)).toBeNull();
        expect(resolveConcreteBuffer(undefined)).toBeNull();
    });
});

// ============================================================================
// compile — single terminal
// ============================================================================

describe('compile — single terminal', () => {
    it('produces same result as topologicalSort for single terminal', () => {
        const buf = new Buffer([1], 'f32');
        const A = makeNode('A', { data: buf });
        const B = makeNode('B', { input: A.data }, [A]);
        const C = makeNode('C', { input: B.input }, [B]);

        const plan = compile([invocation(C)]);

        expect(plan.sorted).toHaveLength(3);
        assertBefore(plan, A, B);
        assertBefore(plan, B, C);
    });

    it('throws on empty terminals', () => {
        expect(() => compile([])).toThrow(/no materialized invocations/);
    });
});

// ============================================================================
// compile — multiple terminals, no shared buffers
// ============================================================================

describe('compile — independent terminals', () => {
    it('accepts several independent terminals from one invocation', () => {
        const buf1 = new Buffer([1], 'f32');
        const buf2 = new Buffer([2], 'f32');

        const A = makeNode('A', { data: buf1 });
        const B = makeNode('B', { data: buf2 });

        const plan = compile([invocation(A, B)]);

        expect(plan.sorted).toHaveLength(2);
        expect(plan.sorted).toContain(A);
        expect(plan.sorted).toContain(B);
    });

    it('merges two independent trees without synthetic deps', () => {
        const buf1 = new Buffer([1], 'f32');
        const buf2 = new Buffer([2], 'f32');

        const A = makeNode('A', { data: buf1 });
        const B = makeNode('B', { data: buf2 });

        const plan = compile([invocation(A), invocation(B)]);

        // Both nodes present, either order is valid
        expect(plan.sorted).toHaveLength(2);
        expect(plan.sorted.map((n) => n._id)).toContain(A._id);
        expect(plan.sorted.map((n) => n._id)).toContain(B._id);
    });

    it('merges two independent chains without synthetic deps', () => {
        const buf1 = new Buffer([1], 'f32');
        const buf2 = new Buffer([2], 'f32');
        const buf3 = new Buffer([3], 'f32');
        const buf4 = new Buffer([4], 'f32');

        const A = makeNode('A', { data: buf1 });
        const B = makeNode('B', { input: A.data, out: buf2 }, [A]);

        const C = makeNode('C', { data: buf3 });
        const D = makeNode('D', { input: C.data, out: buf4 }, [C]);

        const plan = compile([invocation(B), invocation(D)]);

        expect(plan.sorted).toHaveLength(4);
        assertBefore(plan, A, B);
        assertBefore(plan, C, D);
    });
});

// ============================================================================
// compile — shared buffer, synthetic dep injection
// ============================================================================

describe('compile — synthetic dependency injection', () => {
    it('injects synthetic dep when independent terminals share a buffer', () => {
        // The core scenario:
        //   E = k2({ inout });
        //   K = k6({ input: inout });
        //   L = k6({ input: K.input });
        //   v.run([E, L]);
        //
        // E and K both use the same concrete buffer (inout) but have
        // no Handle connection. E must come before K.
        const inout = new Buffer([1], 'f32', 'rw');

        const E = makeNode('E', { inout });
        const K = makeNode('K', { input: inout });
        const L = makeNode('L', { src: K.input }, [K]);

        const plan = compile([invocation(E), invocation(L)]);

        expect(plan.sorted).toHaveLength(3);
        // E must come before K (synthetic dep due to shared buffer)
        assertBefore(plan, E, K);
        // K must come before L (Handle-based dep)
        assertBefore(plan, K, L);
    });

    it('respects positional order: first terminal wins for shared buffers', () => {
        const shared = new Buffer([1], 'f32', 'rw');

        const A = makeNode('A', { buf: shared });
        const B = makeNode('B', { buf: shared });

        // v.run(A, B) → A before B
        const plan1 = compile([invocation(A), invocation(B)]);
        assertBefore(plan1, A, B);

        // v.run(B, A) → B before A
        const plan2 = compile([invocation(B), invocation(A)]);
        assertBefore(plan2, B, A);
    });

    it('chains three terminals with shared buffer', () => {
        // v.run(E, F, G) where all use the same buffer
        const shared = new Buffer([1], 'f32', 'rw');

        const E = makeNode('E', { data: shared });
        const F = makeNode('F', { data: shared });
        const G = makeNode('G', { data: shared });

        const plan = compile([invocation(E), invocation(F), invocation(G)]);

        expect(plan.sorted).toHaveLength(3);
        assertBefore(plan, E, F);
        assertBefore(plan, F, G);
    });

    it('only injects deps for shared buffers, leaves independent buffers alone', () => {
        // E uses bufA
        // F uses bufB (independent)
        // G uses bufA (conflicts with E, but not F)
        const bufA = new Buffer([1], 'f32', 'rw');
        const bufB = new Buffer([2], 'f32', 'rw');

        const E = makeNode('E', { data: bufA });
        const F = makeNode('F', { data: bufB });
        const G = makeNode('G', { data: bufA });

        const plan = compile([invocation(E), invocation(F), invocation(G)]);

        expect(plan.sorted).toHaveLength(3);
        // E must come before G (shared bufA)
        assertBefore(plan, E, G);
        // F has no ordering constraint with E or G (different buffer)
        // Just check all three are present
        expect(plan.sorted.map((n) => n._id)).toContain(F._id);
    });
});

// ============================================================================
// compile — overlapping subtrees (shared nodes via Handles)
// ============================================================================

describe('compile — overlapping subtrees', () => {
    it('handles the fork pattern: D and G both depend on C', () => {
        //  A → B → C → D
        //               C → G
        // v.run(D, G)
        const buf1 = new Buffer([1], 'f32');
        const buf2 = new Buffer([2], 'f32');
        const buf3 = new Buffer([3], 'f32');
        const buf4 = new Buffer([4], 'f32');
        const buf5 = new Buffer([5], 'f32');

        const A = makeNode('A', { data: buf1 });
        const B = makeNode('B', { input: A.data, out: buf2 }, [A]);
        const C = makeNode('C', { input: B.out, out: buf3 }, [B]);
        const D = makeNode('D', { input: C.out, out: buf4 }, [C]);
        const G = makeNode('G', { input: C.out, out: buf5 }, [C]);

        const plan = compile([invocation(D), invocation(G)]);

        // All 5 nodes present (C deduplicated)
        expect(plan.sorted).toHaveLength(5);
        assertBefore(plan, A, B);
        assertBefore(plan, B, C);
        assertBefore(plan, C, D);
        assertBefore(plan, C, G);

        // D and G should NOT have a forced ordering (they're independent after C)
        // Both just need to be after C
    });

    it('deduplicates shared nodes — C executes once', () => {
        const buf1 = new Buffer([1], 'f32');
        const buf2 = new Buffer([2], 'f32');
        const buf3 = new Buffer([3], 'f32');

        const A = makeNode('A', { data: buf1 });
        const B = makeNode('B', { input: A.data, out: buf2 }, [A]);
        const C = makeNode('C', { input: B.out, out: buf3 }, [B]);

        // Two terminals that share the A→B→C subtree
        const D = makeNode('D', { input: C.out }, [C]);
        const G = makeNode('G', { input: C.out }, [C]);

        const plan = compile([invocation(D), invocation(G)]);

        // Exactly 5 unique nodes (not 8 from two separate A→B→C chains)
        expect(plan.sorted).toHaveLength(5);

        // Each node ID appears exactly once
        const ids = plan.sorted.map((n) => n._id);
        expect(new Set(ids).size).toBe(5);
    });
});

// ============================================================================
// compile — inverted positional order
// ============================================================================

describe('compile — inverted argument order', () => {
    it('v.run(D, A) where A is in D subtree produces correct order', () => {
        const buf1 = new Buffer([1], 'f32');
        const buf2 = new Buffer([2], 'f32');

        const A = makeNode('A', { data: buf1 });
        const B = makeNode('B', { input: A.data, out: buf2 }, [A]);

        // v.run(B, A) — A is already in B's subtree
        const plan = compile([invocation(B), invocation(A)]);

        // A still comes before B (Handle-based dep overrides positional order)
        assertBefore(plan, A, B);
    });
});

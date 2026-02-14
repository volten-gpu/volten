import { describe, it, expect } from 'vitest';
import { topologicalSort } from '../src/graph/scheduler.js';
import type { Node } from '../src/graph/node.js';

// Helper to create minimal nodes for testing
function makeNode(idStr: string, dependencies: Node[] = []): Node {
    return {
        _id: Symbol(idStr),
        _dependencies: dependencies,
        // Other properties required by Node interface but unused by scheduler
        _kernel: {} as any,
        _pipeline: {} as any,
        _bindGroupLayout: {} as any,
        _bindingEntries: [],
        _dispatch: [1, 1, 1],
        _bindings: {},
        _shaderCode: '',
    } as any as Node;
}

describe('DAG Scheduler (Kahn\'s Algorithm)', () => {
    it('sorts a linear chain', () => {
        // A -> B -> C
        const A = makeNode('A');
        const B = makeNode('B', [A]);
        const C = makeNode('C', [B]);

        const sorted = topologicalSort(C);

        // Expect: A, B, C
        expect(sorted).toHaveLength(3);
        expect(sorted[0]).toBe(A);
        expect(sorted[1]).toBe(B);
        expect(sorted[2]).toBe(C);
    });

    it('sorts a diamond graph', () => {
        //   /-> B -\
        // A         -> D
        //   \-> C -/
        const A = makeNode('A');
        const B = makeNode('B', [A]);
        const C = makeNode('C', [A]);
        const D = makeNode('D', [B, C]);

        const sorted = topologicalSort(D);

        expect(sorted).toHaveLength(4);
        expect(sorted[0]).toBe(A);
        expect(sorted).toContain(B);
        expect(sorted).toContain(C);
        expect(sorted[3]).toBe(D);

        // B and C can be in any order, but must be after A and before D
        const idxA = sorted.indexOf(A);
        const idxB = sorted.indexOf(B);
        const idxC = sorted.indexOf(C);
        const idxD = sorted.indexOf(D);

        expect(idxA).toBeLessThan(idxB);
        expect(idxA).toBeLessThan(idxC);
        expect(idxB).toBeLessThan(idxD);
        expect(idxC).toBeLessThan(idxD);
    });

    it('sorts complex dependencies', () => {
        // A -> B -> D
        // A -> C -> D
        // E -> C
        const E = makeNode('E');
        const A = makeNode('A');
        const B = makeNode('B', [A]);
        const C = makeNode('C', [A, E]);
        const D = makeNode('D', [B, C]);

        const sorted = topologicalSort(D);

        expect(sorted).toHaveLength(5);
        expect(sorted[4]).toBe(D);

        // Check dependencies
        const idxE = sorted.indexOf(E);
        const idxA = sorted.indexOf(A);
        const idxB = sorted.indexOf(B);
        const idxC = sorted.indexOf(C);
        const idxD = sorted.indexOf(D);

        expect(idxA).toBeLessThan(idxB);
        expect(idxA).toBeLessThan(idxC);
        expect(idxE).toBeLessThan(idxC);
        expect(idxB).toBeLessThan(idxD);
        expect(idxC).toBeLessThan(idxD);
    });

    it('throws on direct cycle', () => {
        // A -> A
        const A = makeNode('A');
        (A as any)._dependencies = [A]; // Hack to create cycle

        expect(() => topologicalSort(A)).toThrow(/Cycle detected/);
    });

    it('throws on indirect cycle', () => {
        // A -> B -> C -> A
        const A = makeNode('A');
        const B = makeNode('B', [A]);
        const C = makeNode('C', [B]);
        (A as any)._dependencies = [C];

        expect(() => topologicalSort(C)).toThrow(/Cycle detected/);
    });

    it('ignores unreachable nodes', () => {
        // A -> B
        // C (unreachable from B)
        const A = makeNode('A');
        const B = makeNode('B', [A]);
        const C = makeNode('C'); // Executing B should not verify C

        const sorted = topologicalSort(B);

        expect(sorted).toHaveLength(2);
        expect(sorted).toContain(A);
        expect(sorted).toContain(B);
        expect(sorted).not.toContain(C);
    });
});

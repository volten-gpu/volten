import { describe, it, expect } from 'vitest';
import { array, struct } from '../src/types/schema.js';
import { getStride, pack } from '../src/utils/alignment.js';
import { Uniform } from '../src/data/uniform.js';

describe('Uniform Layout Rules', () => {
    it('keeps scalar stride unchanged in classic uniform layout', () => {
        expect(getStride('f32', { layoutRules: 'uniform-classic' })).toBe(4);
        expect(getStride('u32', { layoutRules: 'uniform-classic' })).toBe(4);
    });

    it('uses 16-byte array element alignment in classic uniform layout', () => {
        const A = array('f32', 2);

        // Storage layout: stride 4 for f32 array elements, total 8 bytes.
        const storagePacked = pack([[1, 2]], A, { layoutRules: 'storage' });
        expect(storagePacked.byteLength).toBe(8);
        expect(Array.from(new Float32Array(storagePacked).slice(0, 2))).toEqual(
            [1, 2]
        );

        // Classic uniform layout: f32 array elements aligned to 16-byte boundaries.
        const uniformPacked = pack([[1, 2]], A, {
            layoutRules: 'uniform-classic'
        });
        expect(uniformPacked.byteLength).toBe(32);

        const floats = new Float32Array(uniformPacked);
        expect(floats[0]).toBe(1);
        expect(floats[4]).toBe(2); // 16-byte stride => index 4
    });

    it('applies struct-member spacing rule in uniform layout', () => {
        const S = struct('S', {
            x: 'f32'
        });
        const Outer = struct('Outer', {
            a: S,
            b: 'f32'
        });

        const storagePacked = pack([{ a: { x: 10 }, b: 20 }], Outer, {
            layoutRules: 'storage'
        });
        const storageFloats = new Float32Array(storagePacked);
        expect(storagePacked.byteLength).toBe(8);
        expect(storageFloats[0]).toBe(10);
        expect(storageFloats[1]).toBe(20); // offset 4 bytes

        const uniformPacked = pack([{ a: { x: 10 }, b: 20 }], Outer, {
            layoutRules: 'uniform-classic'
        });
        const uniformFloats = new Float32Array(uniformPacked);
        expect(uniformPacked.byteLength).toBe(32);
        expect(uniformFloats[0]).toBe(10);
        expect(uniformFloats[4]).toBe(20); // offset 16 bytes
    });

    it('Uniform uses uniform-aware layout calculations', () => {
        const S = struct('S', {
            x: 'f32'
        });
        const Outer = struct('Outer', {
            a: S,
            b: 'f32'
        });

        const u = new Uniform({ a: { x: 3 }, b: 7 }, Outer);
        expect(u.byteLength).toBe(32);

        const floats = new Float32Array(u.rawData);
        expect(floats[0]).toBe(3);
        expect(floats[4]).toBe(7);
    });
});

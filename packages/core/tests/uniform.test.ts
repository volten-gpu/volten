import { describe, it, expect, vi } from 'vitest';
import { Uniform } from '../src/data/uniform.js';
import { struct } from '../src/types/schema.js';

const GPUBufferUsage = {
    MAP_READ: 0x0001,
    MAP_WRITE: 0x0002,
    COPY_SRC: 0x0004,
    COPY_DST: 0x0008,
    INDEX: 0x0010,
    VERTEX: 0x0020,
    UNIFORM: 0x0040,
    STORAGE: 0x0080,
    INDIRECT: 0x0100,
    QUERY_RESOLVE: 0x0200,
};

(global as any).GPUBufferUsage = GPUBufferUsage;

describe('Uniform', () => {
    it('builds a scalar uniform with expected WGSL type', () => {
        const u = new Uniform(2.0, 'f32');
        expect(u.wgslType).toBe('f32');
        expect(u.byteLength).toBe(4);
    });

    it('supports struct uniforms', () => {
        const Params = struct('Params', {
            offset: 'vec3f',
            scale: 'f32',
        });

        const u = new Uniform(
            { offset: [1, 2, 3], scale: 4.0 },
            Params
        );

        expect(u.wgslType).toBe('Params');
        expect(u.byteLength).toBe(16);
        expect(u.rawData.byteLength).toBe(16);
    });

    it('set() updates local packed data', () => {
        const u = new Uniform(1.0, 'f32');
        const before = new Float32Array(u.rawData)[0];
        u.set(5.0);
        const after = new Float32Array(u.rawData)[0];

        expect(before).toBe(1.0);
        expect(after).toBe(5.0);
    });

    it('set() writes to GPU queue after ensure()', () => {
        const writeBuffer = vi.fn();
        const getMappedRange = vi.fn().mockReturnValue(new ArrayBuffer(4));
        const unmap = vi.fn();
        const destroy = vi.fn();

        const device = {
            queue: {
                writeBuffer,
            },
            createBuffer: vi.fn().mockReturnValue({
                getMappedRange,
                unmap,
                destroy,
            }),
        } as any as GPUDevice;

        const u = new Uniform(1.0, 'f32');
        u.ensure(device);
        u.set(9.0);

        expect(writeBuffer).toHaveBeenCalledTimes(1);
    });

    it('supports standard uniform layout mode packing', () => {
        const S = struct('UniformStdS', { x: 'f32' });
        const Params = struct('UniformStdParams', {
            a: S,
            b: 'f32',
        });

        const u = new Uniform({ a: { x: 3 }, b: 7 }, Params);
        expect(u.byteLength).toBe(32);

        u.setLayoutMode('standard');
        expect(u.byteLength).toBe(8);

        const floats = new Float32Array(u.rawData);
        expect(floats[0]).toBe(3);
        expect(floats[1]).toBe(7);
    });
});

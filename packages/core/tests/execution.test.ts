import { describe, it, expect, vi, beforeEach } from 'vitest';
import { VoltenContext } from '../src/context.js';
import { Kernel } from '../src/kernel/kernel.js';
import { Buffer } from '../src/data/buffer.js';
import { RawBuffer } from '../src/data/raw-buffer.js';
import { Uniform } from '../src/data/uniform.js';

// Mock WebGPU globals for Node.js environment
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
    QUERY_RESOLVE: 0x0200
};
(global as any).GPUBufferUsage = GPUBufferUsage;

const GPUMapMode = {
    READ: 0x0001,
    WRITE: 0x0002
};
(global as any).GPUMapMode = GPUMapMode;

// Mocks
const mockSubmit = vi.fn();
const mockOnSubmittedWorkDone = vi.fn();
const mockWriteBuffer = vi.fn();
const mockCreateBuffer = vi.fn();
const mockCreateBindGroup = vi.fn();
const mockCreateCommandEncoder = vi.fn();
const mockBeginComputePass = vi.fn();
const mockPassSetPipeline = vi.fn();
const mockPassSetBindGroup = vi.fn();
const mockPassDispatch = vi.fn();
const mockPassEnd = vi.fn();
const mockEncoderFinish = vi.fn();
const mockCopyBufferToBuffer = vi.fn();
const mockMapAsync = vi.fn();
const mockUnmap = vi.fn();
const mockDestroy = vi.fn();
const mockGetMappedRange = vi.fn();

const mockDevice = {
    queue: {
        submit: mockSubmit,
        onSubmittedWorkDone: mockOnSubmittedWorkDone,
        writeBuffer: mockWriteBuffer
    },
    createBuffer: mockCreateBuffer,
    createBindGroup: mockCreateBindGroup,
    createCommandEncoder: mockCreateCommandEncoder,
    createShaderModule: vi.fn(),
    createComputePipeline: vi.fn().mockReturnValue({
        getBindGroupLayout: vi.fn()
    })
} as any as GPUDevice;

describe('VoltenContext Execution', () => {
    let v: VoltenContext;

    beforeEach(() => {
        vi.clearAllMocks();
        v = new VoltenContext(mockDevice);

        // Setup mock returns
        mockCreateCommandEncoder.mockReturnValue({
            beginComputePass: mockBeginComputePass,
            finish: mockEncoderFinish,
            copyBufferToBuffer: mockCopyBufferToBuffer
        });
        mockBeginComputePass.mockReturnValue({
            setPipeline: mockPassSetPipeline,
            setBindGroup: mockPassSetBindGroup,
            dispatchWorkgroups: mockPassDispatch,
            end: mockPassEnd
        });
        mockCreateBuffer.mockImplementation((desc: any) => ({
            mapAsync: mockMapAsync,
            getMappedRange:
                (desc.usage & GPUBufferUsage.MAP_READ) !== 0
                    ? mockGetMappedRange
                    : vi.fn(() => new ArrayBuffer(Number(desc.size ?? 16))),
            unmap: mockUnmap,
            destroy: mockDestroy
        }));
        mockGetMappedRange.mockReturnValue(
            new Float32Array([10, 20, 30]).buffer
        );
    });

    it('v.run() executes a linear chain in order', () => {
        // A -> B
        const input = new Buffer([1], 'f32');
        const mid = new Buffer([0], 'f32');
        const output = new Buffer([0], 'f32');

        const K1 = new Kernel('fn main() {}', {
            outputs: { mid: { definedBy: 'input' } },
            threads: 'input'
        });
        const K2 = new Kernel('fn main() {}', {
            outputs: { output: { definedBy: 'data' } },
            threads: 'data'
        });

        const A = v.pass(K1, { input, mid });
        const B = v.pass(K2, { data: A.mid, output });

        v.run(B);

        // Verification
        expect(mockBeginComputePass).toHaveBeenCalledTimes(2); // 2 passes
        expect(mockPassDispatch).toHaveBeenCalledTimes(2);
        expect(mockSubmit).toHaveBeenCalledTimes(1); // 1 submission
    });

    it('v.wait() waits on work done', async () => {
        const input = new Buffer([1], 'f32');
        const output = new Buffer([0], 'f32');
        const K = new Kernel('fn main() {}', {
            outputs: { output: { definedBy: 'input' } },
            threads: 'input'
        });
        const node = v.pass(K, { input, output });

        await v.wait(node);

        expect(mockSubmit).toHaveBeenCalled();
        expect(mockOnSubmittedWorkDone).toHaveBeenCalled();
    });

    it('v.run() supports Uniform bindings and creates a uniform GPU buffer', () => {
        const input = new Buffer([1, 2, 3], 'f32');
        const output = new Buffer([0, 0, 0], 'f32');
        const multiplier = new Uniform(2.0, 'f32');

        const K = new Kernel(
            `
fn main(gid: vec3u) {
    output[gid.x] = input[gid.x] * multiplier;
}
`,
            { threads: 'input' }
        );

        const node = v.pass(K, { input, output, multiplier });
        v.run(node);

        const uniformBufferCall = mockCreateBuffer.mock.calls.find((call) => {
            const desc = call[0];
            return (desc.usage & GPUBufferUsage.UNIFORM) !== 0;
        });

        expect(uniformBufferCall).toBeDefined();
    });

    it('v.run() creates a hidden bounds uniform buffer for guarded kernels', () => {
        const input = new Buffer([1, 2, 3], 'f32');
        const K = new Kernel(
            'fn main(gid: vec3u) { data[gid.x] = data[gid.x]; }',
            {
                threads: 3
            }
        );

        const node = v.pass(K, { data: input });
        v.run(node);

        const uniformBufferCalls = mockCreateBuffer.mock.calls.filter(
            (call) => {
                const desc = call[0];
                return (desc.usage & GPUBufferUsage.UNIFORM) !== 0;
            }
        );

        expect(uniformBufferCalls.length).toBeGreaterThan(0);
    });

    it('v.read() reads back outputs and parses them into TypedArrays without dispatching compute', async () => {
        const input = new Buffer([1], 'f32');
        const outputFloat = new Buffer([0], 'f32');
        const outputUint = new Buffer([0], 'u32');
        const K = new Kernel('fn main() {}', {
            outputs: {
                outputFloat: { type: 'f32', size: 1 },
                outputUint: { type: 'u32', size: 1 }
            },
            threads: 'input'
        });
        const node = v.pass(K, { input, outputFloat, outputUint });

        v.run(node);

        mockSubmit.mockClear();
        mockBeginComputePass.mockClear();
        mockPassDispatch.mockClear();
        mockCopyBufferToBuffer.mockClear();
        mockMapAsync.mockClear();
        mockGetMappedRange.mockClear();

        const result = await v.read(node);

        expect(mockBeginComputePass).not.toHaveBeenCalled();
        expect(mockPassDispatch).not.toHaveBeenCalled();
        expect(mockSubmit).toHaveBeenCalledTimes(1);
        expect(mockCopyBufferToBuffer).toHaveBeenCalled();
        expect(mockMapAsync).toHaveBeenCalled();
        expect(mockGetMappedRange).toHaveBeenCalled();

        // Check result structure
        expect(result).toHaveProperty('outputFloat');
        expect(result['outputFloat']).toBeInstanceOf(Float32Array);

        expect(result).toHaveProperty('outputUint');
        expect(result['outputUint']).toBeInstanceOf(Uint32Array);
    });

    // =================================================================
    // Multi-node v.run()
    // =================================================================

    it('v.run(A, B) executes two independent nodes', () => {
        const buf1 = new Buffer([1], 'f32');
        const buf2 = new Buffer([2], 'f32');

        const K = new Kernel('fn main() {}', { threads: 1 });

        const A = v.pass(K, { data: buf1 });
        const B = v.pass(K, { data: buf2 });

        v.run([A, B]);

        // Both nodes should dispatch
        expect(mockPassDispatch).toHaveBeenCalledTimes(2);
        expect(mockSubmit).toHaveBeenCalledTimes(1);
    });

    it('v.run([A, B]) array form works', () => {
        const buf1 = new Buffer([1], 'f32');
        const buf2 = new Buffer([2], 'f32');

        const K = new Kernel('fn main() {}', { threads: 1 });

        const A = v.pass(K, { data: buf1 });
        const B = v.pass(K, { data: buf2 });

        v.run([A, B]);

        expect(mockPassDispatch).toHaveBeenCalledTimes(2);
        expect(mockSubmit).toHaveBeenCalledTimes(1);
    });

    it('v.run(E, L) with shared buffer dispatches all three nodes', () => {
        // E and K use the same buffer, L chains from K
        const shared = new Buffer([1], 'f32', 'rw');

        const K1 = new Kernel('fn main() {}', { threads: 1 });
        const K2 = new Kernel('fn main() {}', { threads: 1 });

        const E = v.pass(K1, { inout: shared });
        const K = v.pass(K2, { input: shared });
        const L = v.pass(K2, { src: K.input });

        v.run([E, L]);

        // E, K, L = 3 dispatches
        expect(mockPassDispatch).toHaveBeenCalledTimes(3);
        expect(mockSubmit).toHaveBeenCalledTimes(1);
    });

    it('Uniform.set() pushes updates via queue.writeBuffer after upload', () => {
        const uniform = new Uniform(1.0, 'f32');

        // Not uploaded yet: local update only
        uniform.set(2.0);
        expect(mockWriteBuffer).not.toHaveBeenCalled();

        // Upload
        uniform.ensure(mockDevice);

        // Uploaded: should write to GPU queue
        uniform.set(3.0);
        expect(mockWriteBuffer).toHaveBeenCalledTimes(1);
    });

    it('Buffer.set() and update() push updates after upload', () => {
        const buffer = new Buffer([1, 2, 3], 'f32');

        buffer.set([4, 5, 6]);
        expect(mockWriteBuffer).not.toHaveBeenCalled();
        expect(Array.from(new Float32Array(buffer.rawData))).toEqual([4, 5, 6]);

        buffer.ensure(mockDevice);
        buffer.update([9], 1);

        expect(mockWriteBuffer).toHaveBeenCalledTimes(1);
        expect(mockWriteBuffer.mock.calls[0][1]).toBe(4);
        expect(Array.from(new Float32Array(buffer.rawData))).toEqual([4, 9, 6]);
    });

    it('Buffer.set() rejects element count changes', () => {
        const buffer = new Buffer([1, 2, 3], 'f32');

        expect(() => buffer.set([1, 2])).toThrow(
            /cannot resize the underlying GPU buffer/
        );
    });

    it('RawBuffer.set() and update() push byte updates after upload', () => {
        const raw = new RawBuffer(new Uint32Array([1, 2]).buffer, 'array<u32>');

        raw.set(new Uint32Array([3, 4]));
        expect(mockWriteBuffer).not.toHaveBeenCalled();
        expect(Array.from(new Uint32Array(raw.rawData))).toEqual([3, 4]);

        raw.ensure(mockDevice);
        raw.update(new Uint32Array([9]), 4);

        expect(mockWriteBuffer).toHaveBeenCalledTimes(1);
        expect(mockWriteBuffer.mock.calls[0][1]).toBe(4);
        expect(Array.from(new Uint32Array(raw.rawData))).toEqual([3, 9]);
    });

    it('RawBuffer.set() rejects byte length changes', () => {
        const raw = new RawBuffer(new Uint32Array([1, 2]).buffer, 'array<u32>');

        expect(() => raw.set(new Uint32Array([1, 2, 3]))).toThrow(
            /cannot resize the underlying GPU buffer/
        );
    });

    it('reuses a node bind group across repeated runs', () => {
        const data = new Buffer([1], 'f32');
        const K = new Kernel('fn main() {}', {
            threads: 1,
            unsafeManualBounds: true
        });
        const node = v.pass(K, { data });

        v.run(node);
        v.run(node);

        expect(mockCreateBindGroup).toHaveBeenCalledTimes(1);
        expect(mockPassSetBindGroup).toHaveBeenCalledTimes(2);
        expect(mockPassDispatch).toHaveBeenCalledTimes(2);
    });

    it('keeps the cached bind group after a same-size Uniform.set()', () => {
        const params = new Uniform(1.0, 'f32');
        const K = new Kernel('fn main() {}', {
            threads: 1,
            unsafeManualBounds: true
        });
        const node = v.pass(K, { params });

        v.run(node);
        params.set(2.0);
        v.run(node);

        expect(mockWriteBuffer).toHaveBeenCalledTimes(1);
        expect(mockCreateBindGroup).toHaveBeenCalledTimes(1);
    });

    it('keeps the cached bind group after a same-size Buffer.set()', () => {
        const data = new Buffer([1], 'f32');
        const K = new Kernel('fn main() {}', {
            threads: 1,
            unsafeManualBounds: true
        });
        const node = v.pass(K, { data });

        v.run(node);
        data.set([2]);
        v.run(node);

        expect(mockWriteBuffer).toHaveBeenCalledTimes(1);
        expect(mockCreateBindGroup).toHaveBeenCalledTimes(1);
    });

    it('recreates a cached bind group after the GPU buffer changes', () => {
        const data = new Buffer([1], 'f32');
        const K = new Kernel('fn main() {}', {
            threads: 1,
            unsafeManualBounds: true
        });
        const node = v.pass(K, { data });

        v.run(node);
        data.destroy();
        v.run(node);

        expect(mockDestroy).toHaveBeenCalledTimes(1);
        expect(mockCreateBindGroup).toHaveBeenCalledTimes(2);
    });

    it('v.destroy() recursively destroys only Volten-owned internal resources', () => {
        const input = new Buffer([1], 'f32');
        const mid = new Buffer([0], 'f32');
        const output = new Buffer([0], 'f32');
        const multiplier = new Uniform(2.0, 'f32');

        const K1 = new Kernel('fn main() {}', {
            outputs: { mid: { definedBy: 'input' } },
            threads: 'input'
        });
        const K2 = new Kernel('fn main() {}', {
            outputs: { output: { definedBy: 'data' } },
            threads: 'data'
        });

        const A = v.pass(K1, { input, mid });
        const B = v.pass(K2, { data: A.mid, output, multiplier });

        v.run(B);

        expect(multiplier.isUploaded).toBe(true);

        v.destroy(B);

        expect(mockDestroy).toHaveBeenCalledTimes(2);
        expect(multiplier.isUploaded).toBe(true);
    });

    it('v.destroy() is idempotent', () => {
        const input = new Buffer([1], 'f32');
        const K = new Kernel('fn main() {}', { threads: 'data' });
        const node = v.pass(K, { data: input });

        v.run(node);
        v.destroy(node);

        expect(mockDestroy).toHaveBeenCalledTimes(1);

        v.destroy(node);

        expect(mockDestroy).toHaveBeenCalledTimes(1);
    });

    it('v.destroy() allows reruns by recreating internal bounds resources', () => {
        const input = new Buffer([1, 2, 3], 'f32');
        const K = new Kernel(
            'fn main(gid: vec3u) { data[gid.x] = data[gid.x]; }',
            {
                threads: 3
            }
        );
        const node = v.pass(K, { data: input });

        v.run(node);

        const initialUniformBufferCreations =
            mockCreateBuffer.mock.calls.filter(
                (call) => (call[0].usage & GPUBufferUsage.UNIFORM) !== 0
            ).length;

        expect(initialUniformBufferCreations).toBe(1);

        v.destroy(node);
        v.run(node);

        const totalUniformBufferCreations = mockCreateBuffer.mock.calls.filter(
            (call) => (call[0].usage & GPUBufferUsage.UNIFORM) !== 0
        ).length;

        expect(mockDestroy).toHaveBeenCalledTimes(1);
        expect(totalUniformBufferCreations).toBe(2);
        expect(mockCreateBindGroup).toHaveBeenCalledTimes(2);
    });
});

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { VoltenContext } from '../src/context.js';
import { Kernel } from '../src/kernel/kernel.js';
import { Buffer } from '../src/data/buffer.js';

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
    QUERY_RESOLVE: 0x0200,
};
(global as any).GPUBufferUsage = GPUBufferUsage;

const GPUMapMode = {
    READ: 0x0001,
    WRITE: 0x0002,
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
        writeBuffer: mockWriteBuffer,
    },
    createBuffer: mockCreateBuffer,
    createBindGroup: mockCreateBindGroup,
    createCommandEncoder: mockCreateCommandEncoder,
    createShaderModule: vi.fn(),
    createComputePipeline: vi.fn().mockReturnValue({
        getBindGroupLayout: vi.fn(),
    }),
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
            copyBufferToBuffer: mockCopyBufferToBuffer,
        });
        mockBeginComputePass.mockReturnValue({
            setPipeline: mockPassSetPipeline,
            setBindGroup: mockPassSetBindGroup,
            dispatchWorkgroups: mockPassDispatch,
            end: mockPassEnd,
        });
        mockCreateBuffer.mockReturnValue({
            mapAsync: mockMapAsync,
            getMappedRange: mockGetMappedRange,
            unmap: mockUnmap,
            destroy: mockDestroy,
        });
        mockGetMappedRange.mockReturnValue(new Float32Array([10, 20, 30]).buffer);
    });

    it('v.run() executes a linear chain in order', () => {
        // A -> B
        const input = new Buffer([1], 'f32');
        const mid = new Buffer([0], 'f32');
        const output = new Buffer([0], 'f32');

        const K1 = new Kernel('fn main() {}', { outputs: { mid: { definedBy: 'input' } }, threads: 'input' });
        const K2 = new Kernel('fn main() {}', { outputs: { output: { definedBy: 'data' } }, threads: 'data' });

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
        const K = new Kernel('fn main() {}', { outputs: { output: { definedBy: 'input' } }, threads: 'input' });
        const node = v.pass(K, { input, output });

        await v.wait(node);

        expect(mockSubmit).toHaveBeenCalled();
        expect(mockOnSubmittedWorkDone).toHaveBeenCalled();
    });

    it('v.read() reads back outputs', async () => {
        const input = new Buffer([1], 'f32');
        const output = new Buffer([0], 'f32');
        const K = new Kernel('fn main() {}', { outputs: { output: { definedBy: 'input' } }, threads: 'input' });
        const node = v.pass(K, { input, output });

        const result = await v.read(node);

        expect(mockSubmit).toHaveBeenCalled(); // Compute + Copy (might be 1 or 2 submits depending on implementation)
        // Check copy
        expect(mockCopyBufferToBuffer).toHaveBeenCalled();
        // Check map
        expect(mockMapAsync).toHaveBeenCalled();
        expect(mockGetMappedRange).toHaveBeenCalled();

        // Check result structure
        expect(result).toHaveProperty('output');
        expect(result['output']).toBeInstanceOf(Float32Array);
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

        v.run(A, B);

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

        v.run(E, L);

        // E, K, L = 3 dispatches
        expect(mockPassDispatch).toHaveBeenCalledTimes(3);
        expect(mockSubmit).toHaveBeenCalledTimes(1);
    });
});


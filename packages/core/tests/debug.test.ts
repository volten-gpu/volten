import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Kernel } from '../src/kernel/kernel.js';
import { prepareKernelShader } from '../src/kernel/shader.js';
import { VoltenContext } from '../src/context.js';
import {
    VOLTEN_DEBUG_BUFFER_NAME,
    createDebugTransform
} from '../src/debug/index.js';

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

function encodeF32(value: number): number {
    const view = new DataView(new ArrayBuffer(4));
    view.setFloat32(0, value, true);
    return view.getUint32(0, true);
}

function createMockPassContext(): VoltenContext {
    const mockBindGroupLayout = { label: 'mock-layout' };
    const mockPipeline = {
        label: 'mock-pipeline',
        getBindGroupLayout: vi.fn().mockReturnValue(mockBindGroupLayout)
    };
    const mockDevice = {
        createShaderModule: vi.fn().mockReturnValue({ label: 'mock-module' }),
        createComputePipeline: vi.fn().mockReturnValue(mockPipeline)
    } as any;

    return new VoltenContext(mockDevice);
}

describe('Debug Shader Preparation', () => {
    it('keeps debug helper names readable while rewriting message strings', () => {
        const kernel = new Kernel(
            `
fn main() {
    enableDebug();
    debugF32("after multiply", 3.5);
    debugVec3("normal", vec3f(1.0, 2.0, 3.0));
}
`,
            { threads: 1 }
        );

        const debug = createDebugTransform(256);
        const prepared = prepareKernelShader(kernel, {
            transforms: [debug]
        });

        expect(debug.messages).toEqual(['after multiply', 'normal']);
        expect(prepared.kernelSource).toContain(
            '@builtin(global_invocation_id) _volten_guard_gid_builtin: vec3<u32>'
        );
        expect(prepared.kernelSource).toContain(
            '_volten_debug_begin_invocation(_volten_guard_gid_builtin);'
        );
        expect(prepared.kernelSource).toContain('enableDebug();');
        expect(prepared.kernelSource).toContain('debugF32(1u, 3.5)');
        expect(prepared.kernelSource).toContain(
            'debugVec3(2u, vec3f(1.0, 2.0, 3.0))'
        );
        const supportWgsl = prepared.supportWgsl.join('\n');
        expect(supportWgsl).toContain('fn enableDebug() {');
        expect(supportWgsl).toContain(
            'fn debugF32(messageId: u32, value: f32) {'
        );
        expect(supportWgsl).toContain(
            'var<private> _volten_debug_enabled: bool = false;'
        );
    });

    it('throws on unsupported generic debug(...) syntax', () => {
        const kernel = new Kernel(
            `
fn main() {
    debug("f32", "value", 1.0);
}
`,
            { threads: 1 }
        );

        const debug = createDebugTransform(256);
        expect(() =>
            prepareKernelShader(kernel, { transforms: [debug] })
        ).toThrow(/Generic debug\(\.\.\.\) is not supported/);
    });

    it('keeps comparison operators from affecting debug argument splitting', () => {
        const kernel = new Kernel(
            `
fn choose(a: f32, b: f32, useA: bool) -> f32 {
    return select(b, a, useA);
}

fn main() {
    debugF32("chosen", choose(1.0, 2.0, 3.0 > 2.0));
}
`,
            { threads: 1 }
        );

        const debug = createDebugTransform(256);
        const prepared = prepareKernelShader(kernel, {
            transforms: [debug]
        });

        expect(debug.messages).toEqual(['chosen']);
        expect(prepared.kernelSource).toContain(
            'debugF32(1u, choose(1.0, 2.0, 3.0 > 2.0))'
        );
    });
});

describe('VoltenContext Debug Integration', () => {
    const mockSubmit = vi.fn();
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
            onSubmittedWorkDone: vi.fn(),
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

    beforeEach(() => {
        vi.clearAllMocks();

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
        mockMapAsync.mockResolvedValue(undefined);
        mockCreateBuffer.mockImplementation((desc: any) => ({
            mapAsync: mockMapAsync,
            getMappedRange:
                (desc.usage & GPUBufferUsage.MAP_READ) !== 0
                    ? mockGetMappedRange
                    : vi.fn(() => new ArrayBuffer(Number(desc.size ?? 16))),
            unmap: mockUnmap,
            destroy: mockDestroy
        }));
    });

    it('injects a hidden debug buffer and keeps it off the public node shape', () => {
        const v = createMockPassContext();
        const kernel = new Kernel(
            `
fn main() {
    enableDebug();
    debugF32("value", 1.0);
}
`,
            { threads: 1 }
        );

        const node = v.pass(kernel, {}, { debug: true });

        expect(
            node._bindingEntries.find(
                (entry) => entry.name === VOLTEN_DEBUG_BUFFER_NAME
            )
        ).toBeDefined();
        expect(node._debug).not.toBeNull();
        expect((node as any)[VOLTEN_DEBUG_BUFFER_NAME]).toBeUndefined();
        expect(node._shaderCode).toContain(
            `var<storage, read_write> ${VOLTEN_DEBUG_BUFFER_NAME}: _volten_debug_storage_buffer;`
        );
    });

    it('resets the hidden debug header before dispatch', () => {
        const v = new VoltenContext(mockDevice);
        const kernel = new Kernel(
            `
fn main() {
    enableDebug();
    debugF32("value", 1.0);
}
`,
            { threads: 1 }
        );

        const node = v.pass(kernel, {}, { debug: true });
        v.run(node);

        expect(mockWriteBuffer).toHaveBeenCalledTimes(1);
        expect(Array.from(mockWriteBuffer.mock.calls[0][2] as Uint32Array)).toEqual(
            [0, 0]
        );
    });

    it('decodes debug logs from the hidden debug buffer', async () => {
        const v = new VoltenContext(mockDevice);
        const kernel = new Kernel(
            `
fn main() {
    enableDebug();
    debugF32("scalar", 1.0);
    debugVec3("vector", vec3f(1.0, 2.0, 3.0));
}
`,
            { threads: 1 }
        );

        const node = v.pass(kernel, {}, { debug: { bufferSize: 128 } });

        const totalWords = node._debug!.resource.buffer.byteLength / 4;
        const words = new Uint32Array(totalWords);

        // cursor / dropped
        words[0] = 16;
        words[1] = 0;

        // Record 1: f32
        words[2] = 1;
        words[3] = 1;
        words[4] = 57;
        words[5] = 0;
        words[6] = 0;
        words[7] = 1;
        words[8] = encodeF32(3.5);

        // Record 2: vec3f
        words[9] = 5;
        words[10] = 2;
        words[11] = 1;
        words[12] = 2;
        words[13] = 3;
        words[14] = 3;
        words[15] = encodeF32(1.0);
        words[16] = encodeF32(2.0);
        words[17] = encodeF32(3.0);

        mockGetMappedRange.mockReturnValue(words.buffer);

        const result = await v.readDebug(node);

        expect(result.dropped).toBe(0);
        expect(result.logs).toHaveLength(2);
        expect(result.logs[0]).toEqual({
            kind: 'f32',
            gid: [57, 0, 0],
            message: 'scalar',
            value: 3.5
        });
        expect(result.logs[1]).toEqual({
            kind: 'vec3f',
            gid: [1, 2, 3],
            message: 'vector',
            value: [1, 2, 3]
        });
    });

    it('throws when readDebug() is called on a node without debug support', async () => {
        const v = new VoltenContext(mockDevice);
        const kernel = new Kernel('fn main() { }', { threads: 1 });
        const node = v.pass(kernel, {});

        await expect(v.readDebug(node)).rejects.toThrow(
            /without debug support/
        );
    });
});

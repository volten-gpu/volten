/**
 * Tests for binding generation, shader assembly, dispatch resolution,
 * physical node creation, and callable-kernel materialization.
 *
 * Since we're in a Node.js environment without a real GPU, tests cover:
 * 1. Binding classification (Buffer, RawBuffer, Handle)
 * 2. WGSL binding declaration generation
 * 3. Full shader assembly
 * 4. Thread dispatch resolution
 * 5. Error handling & validation
 * 6. Callable operation integration with a mocked GPUDevice
 */
import { describe, it, expect, vi } from 'vitest';
import {
    generateBindings,
    generateBindingWgsl,
    assembleFullShader,
    resolveBounds,
    resolveDispatch
} from '../src/kernel/bindings.js';
import { Kernel, kernel } from '../src/kernel/kernel.js';
import { prepareKernelShader } from '../src/kernel/shader.js';
import { VOLTEN_BOUNDS_NAME } from '../src/kernel/builtins.js';
import { Buffer } from '../src/data/buffer.js';
import { RawBuffer } from '../src/data/raw-buffer.js';
import { Uniform } from '../src/data/uniform.js';
import { array, struct } from '../src/types/schema.js';
import {
    createDispatchNode as createNode,
    createDispatchHandle as createHandle,
    validateOutputName,
    getDispatchNodeHandles as getNodeHandles,
    getDispatchNodeOutputHandles as getNodeOutputHandles,
    isBufferLike,
    type DispatchHandle as Handle
} from '../src/graph/dispatch-node.js';
import { PipelineCache } from '../src/graph/pipeline-cache.js';
import { VoltenContext } from '../src/context.js';
import { supportsWgslLanguageFeature } from '../src/utils/wgsl-features.js';
// =============================================================================
// Helper: create buffers for testing (no GPU needed for classification)
// =============================================================================
function makeBuffer(
    data: number[] = [1, 2, 3, 4],
    type: string = 'f32',
    access: 'r' | 'rw' = 'r'
): Buffer {
    return new Buffer(data, type as any, access);
}
function invoke(
    definition: Kernel,
    bindings: Record<string, any> = {},
    options?: any
) {
    const operation = kernel({
        shader: definition.source,
        label: definition.label,
        outputs: definition.outputNames,
        workgroupSize: definition.workgroupSize,
        threads: definition.threads,
        unsafeManualBounds: definition.unsafeManualBounds
    });
    return operation(bindings, options);
}
function getDispatch(v: VoltenContext, node: any): any {
    return (v as any)._materializer.lowerNode(node).terminals[0];
}
function assembleKernelShader(
    kernel: Kernel,
    entries: ReturnType<typeof generateBindings>,
    options?: {
        uniformLayoutMode?: 'classic' | 'standard';
    }
): string {
    const prepared = prepareKernelShader(kernel);
    return assembleFullShader(entries, {
        ...options,
        kernelSource: prepared.kernelSource
    });
}
function makeRawBuffer(
    size: number = 16,
    type: string = 'array<f32>'
): RawBuffer {
    return new RawBuffer(new ArrayBuffer(size), type);
}
function makeUniform(data: unknown = 1.0, type: string = 'f32'): Uniform {
    return new Uniform(data, type as any);
}
// Helper: create a minimal mock Handle backed by a real Buffer
function makeMockHandle(name: string = 'output'): Handle {
    const backingBuffer = makeBuffer([0], 'f32', 'r');
    const fakeNode = {
        _id: Symbol('FakeNode'),
        _label: 'FakeNode',
        _bindings: { [name]: backingBuffer }
    } as any;
    return {
        _id: Symbol(`Handle:${name}`),
        _node: fakeNode,
        _name: name,
        _label: `FakeNode.${name}`
    };
}
// =============================================================================
// BINDING GENERATION TESTS
// =============================================================================
describe('Binding Generation', () => {
    describe('generateBindings()', () => {
        it('classifies a read-only Buffer binding', () => {
            const buf = makeBuffer([1, 2, 3], 'f32', 'r');
            const entries = generateBindings({ input: buf });
            expect(entries).toHaveLength(1);
            expect(entries[0].index).toBe(0);
            expect(entries[0].name).toBe('input');
            expect(entries[0].wgslType).toBe('array<f32>');
            expect(entries[0].wgslAccess).toBe('read');
            expect(entries[0].isHandle).toBe(false);
        });
        it('classifies a read-write Buffer binding', () => {
            const buf = makeBuffer([1, 2, 3], 'f32', 'rw');
            const entries = generateBindings({ output: buf });
            expect(entries[0].wgslAccess).toBe('read_write');
        });
        it('classifies a RawBuffer binding', () => {
            const raw = makeRawBuffer(64, 'array<vec4f>');
            const entries = generateBindings({ data: raw });
            expect(entries).toHaveLength(1);
            expect(entries[0].name).toBe('data');
            expect(entries[0].wgslType).toBe('array<vec4f>');
            expect(entries[0].isHandle).toBe(false);
        });
        it('classifies a Uniform binding', () => {
            const uniform = makeUniform([1, 2, 3, 4], 'vec4f');
            const entries = generateBindings({ transform: uniform });
            expect(entries).toHaveLength(1);
            expect(entries[0].name).toBe('transform');
            expect(entries[0].wgslType).toBe('vec4f');
            expect(entries[0].wgslAddressSpace).toBe('uniform');
            expect(entries[0].wgslAccess).toBeUndefined();
            expect(entries[0].isHandle).toBe(false);
        });
        it('classifies a Handle binding with resolved type from source', () => {
            const handle = makeMockHandle('prevOutput');
            const entries = generateBindings({ prevData: handle });
            expect(entries).toHaveLength(1);
            expect(entries[0].name).toBe('prevData');
            // Resolved from the backing buffer (f32, read-only)
            expect(entries[0].wgslType).toBe('array<f32>');
            expect(entries[0].wgslAccess).toBe('read');
            expect(entries[0].isHandle).toBe(true);
        });
        it('assigns sequential binding indices', () => {
            const a = makeBuffer([1], 'f32', 'r');
            const b = makeBuffer([2], 'f32', 'rw');
            const c = makeRawBuffer(4, 'array<u32>');
            const entries = generateBindings({ a, b, c });
            expect(entries[0].index).toBe(0);
            expect(entries[1].index).toBe(1);
            expect(entries[2].index).toBe(2);
        });
        it('throws on number binding with helpful hint', () => {
            expect(() => generateBindings({ scale: 2.5 as any })).toThrow(
                /Raw scalar bindings are not supported/
            );
            expect(() => generateBindings({ scale: 2.5 as any })).toThrow(
                /new Buffer/
            );
        });
        it('throws on unsupported binding type', () => {
            expect(() => generateBindings({ name: 'hello' as any })).toThrow(
                /unsupported type/
            );
        });
        it('handles empty bindings with no kernel outputs', () => {
            const entries = generateBindings({});
            expect(entries).toHaveLength(0);
        });
    });
});
// =============================================================================
// WGSL BINDING DECLARATION TESTS
// =============================================================================
describe('WGSL Binding Declaration Generation', () => {
    describe('generateBindingWgsl()', () => {
        it('generates correct binding declaration for read-only buffer', () => {
            const buf = makeBuffer([1], 'f32', 'r');
            const entries = generateBindings({ input: buf });
            const wgsl = generateBindingWgsl(entries);
            expect(wgsl).toBe(
                '@group(0) @binding(0) var<storage, read> input: array<f32>;'
            );
        });
        it('generates correct binding declaration for read-write buffer', () => {
            const buf = makeBuffer([1], 'f32', 'rw');
            const entries = generateBindings({ output: buf });
            const wgsl = generateBindingWgsl(entries);
            expect(wgsl).toBe(
                '@group(0) @binding(0) var<storage, read_write> output: array<f32>;'
            );
        });
        it('generates multiple binding declarations', () => {
            const input = makeBuffer([1], 'f32', 'r');
            const output = makeBuffer([1], 'f32', 'rw');
            const entries = generateBindings({ input, output });
            const wgsl = generateBindingWgsl(entries);
            const lines = wgsl.split('\n');
            expect(lines).toHaveLength(2);
            expect(lines[0]).toContain('@binding(0)');
            expect(lines[0]).toContain('input');
            expect(lines[1]).toContain('@binding(1)');
            expect(lines[1]).toContain('output');
        });
        it('returns empty string for no bindings', () => {
            expect(generateBindingWgsl([])).toBe('');
        });
        it('generates var<uniform> declaration for Uniform bindings', () => {
            const scale = makeUniform(2.0, 'f32');
            const entries = generateBindings({ scale });
            const wgsl = generateBindingWgsl(entries);
            expect(wgsl).toBe('@group(0) @binding(0) var<uniform> scale: f32;');
        });
        it('handles vec3f buffer type', () => {
            const buf = new Buffer([[1, 2, 3]], 'vec3f' as any, 'r');
            const entries = generateBindings({ positions: buf });
            const wgsl = generateBindingWgsl(entries);
            expect(wgsl).toContain('array<vec3f>');
        });
    });
});
// =============================================================================
// SHADER ASSEMBLY TESTS
// =============================================================================
describe('Shader Assembly', () => {
    describe('assembleFullShader()', () => {
        it('prepends binding declarations to kernel source', () => {
            const kernel = new Kernel(`
fn main(gid: vec3u) {
    output[gid.x] = input[gid.x] * 2.0;
}
`);
            const input = makeBuffer([1, 2, 3], 'f32', 'r');
            const output = makeBuffer([0, 0, 0], 'f32', 'rw');
            const entries = generateBindings({ input, output });
            const shader = assembleKernelShader(kernel, entries);
            // Bindings should come before the function
            const bindingPos = shader.indexOf('@group(0)');
            const fnPos = shader.indexOf('fn main');
            expect(bindingPos).toBeLessThan(fnPos);
            expect(bindingPos).toBeGreaterThanOrEqual(0);
        });
        it('includes compute decorators from kernel assembly', () => {
            const kernel = new Kernel('fn main(gid: vec3u) { }', {
                workgroupSize: [128]
            });
            const entries = generateBindings({});
            const shader = assembleKernelShader(kernel, entries);
            expect(shader).toContain('@compute @workgroup_size(128, 1, 1)');
        });
        it('includes builtin expansion from kernel assembly', () => {
            const kernel = new Kernel('fn main(gid: vec3u) { }');
            const entries = generateBindings({});
            const shader = assembleKernelShader(kernel, entries);
            expect(shader).toContain(
                '@builtin(global_invocation_id) gid: vec3u'
            );
        });
        it('returns kernel source alone when no bindings', () => {
            const kernel = new Kernel('fn main() { }');
            const entries = generateBindings({});
            const shader = assembleKernelShader(kernel, entries);
            expect(shader).toBe(prepareKernelShader(kernel).kernelSource);
        });
        it('auto-emits struct declarations for struct-typed storage bindings', () => {
            const kernel = new Kernel('fn main() { }');
            const Particle = struct('Particle', {
                mass: 'f32'
            });
            const particles = new Buffer([{ mass: 1 }], Particle, 'r');
            const entries = generateBindings({ particles });
            const shader = assembleKernelShader(kernel, entries);
            expect(shader).toContain('struct Particle {');
            expect(shader).toContain('mass: f32,');
            expect(shader).toContain(
                'var<storage, read> particles: array<Particle>;'
            );
        });
        it('emits classic uniform alignment attributes for nested-struct spacing', () => {
            const kernel = new Kernel('fn main() { }');
            const S = struct('S', { x: 'f32' });
            const Params = struct('Params', {
                a: S,
                b: 'f32'
            });
            const params = new Uniform({ a: { x: 1 }, b: 2 }, Params);
            const entries = generateBindings(
                { params },
                {
                    uniformLayoutMode: 'classic'
                }
            );
            const shader = assembleKernelShader(kernel, entries, {
                uniformLayoutMode: 'classic'
            });
            expect(shader).toContain('struct Params {');
            expect(shader).toContain('@align(16) b: f32,');
        });
        it('emits extension requirement in standard uniform mode', () => {
            const kernel = new Kernel('fn main() { }');
            const S = struct('SStd', { x: 'f32' });
            const Params = struct('ParamsStd', {
                a: S,
                b: 'f32'
            });
            const params = new Uniform({ a: { x: 1 }, b: 2 }, Params);
            const entries = generateBindings(
                { params },
                {
                    uniformLayoutMode: 'standard'
                }
            );
            const shader = assembleKernelShader(kernel, entries, {
                uniformLayoutMode: 'standard'
            });
            expect(params.byteLength).toBe(8);
            expect(shader).toContain(
                'requires uniform_buffer_standard_layout;'
            );
            expect(shader).not.toContain('@align(16) b: f32,');
        });
        it('throws for classic uniform arrays that require wrapper element types', () => {
            const kernel = new Kernel('fn main() { }');
            const Params = struct('ArrayParams', {
                weights: array('f32', 4)
            });
            const params = new Uniform({ weights: [1, 2, 3, 4] }, Params);
            const entries = generateBindings(
                { params },
                {
                    uniformLayoutMode: 'classic'
                }
            );
            expect(() =>
                assembleKernelShader(kernel, entries, {
                    uniformLayoutMode: 'classic'
                })
            ).toThrow(/16-byte array stride/);
        });
        it('throws when one struct name is shared by storage and classic-uniform variants', () => {
            const kernel = new Kernel('fn main() { }');
            const SharedS = struct('SharedS', { x: 'f32' });
            const SharedParams = struct('SharedParams', {
                a: SharedS,
                b: 'f32'
            });
            const storageData = new Buffer(
                [{ a: { x: 1 }, b: 2 }],
                SharedParams,
                'r'
            );
            const uniformData = new Uniform(
                { a: { x: 1 }, b: 2 },
                SharedParams
            );
            const entries = generateBindings(
                { storageData, uniformData },
                { uniformLayoutMode: 'classic' }
            );
            expect(() =>
                assembleKernelShader(kernel, entries, {
                    uniformLayoutMode: 'classic'
                })
            ).toThrow(/used by both storage and classic-uniform layouts/);
        });
    });
});
// =============================================================================
// THREAD DISPATCH RESOLUTION TESTS
// =============================================================================
describe('Thread Dispatch Resolution', () => {
    describe('resolveBounds()', () => {
        it('resolves logical bounds before workgroup division', () => {
            const kernel = new Kernel('fn main() { }', {
                workgroupSize: [64, 4, 1]
            });
            expect(resolveBounds(kernel, {}, [130, 5, 2])).toEqual([130, 5, 2]);
        });
        it('normalizes a single-element pass-time thread tuple to 1D bounds', () => {
            const kernel = new Kernel('fn main() { }');
            expect(resolveBounds(kernel, {}, [130])).toEqual([130, 1, 1]);
        });
    });
    describe('resolveDispatch()', () => {
        it('resolves numeric threads spec', () => {
            const kernel = new Kernel('fn main() { }', { threads: 256 });
            const dispatch = resolveDispatch(kernel, {});
            // 256 threads / 64 workgroup = 4 dispatches
            expect(dispatch).toEqual([4, 1, 1]);
        });
        it('resolves numeric threads with custom workgroup size', () => {
            const kernel = new Kernel('fn main() { }', {
                threads: 1000,
                workgroupSize: [256]
            });
            const dispatch = resolveDispatch(kernel, {});
            // ceil(1000 / 256) = 4
            expect(dispatch).toEqual([4, 1, 1]);
        });
        it('resolves string threads spec (infer from named input)', () => {
            const input = makeBuffer([1, 2, 3, 4, 5, 6, 7, 8], 'f32', 'r');
            const kernel = new Kernel('fn main() { }', { threads: 'input' });
            const dispatch = resolveDispatch(kernel, { input });
            // 8 elements / 64 workgroup = 1 dispatch (ceil)
            expect(dispatch).toEqual([1, 1, 1]);
        });
        it('resolves function threads spec', () => {
            const kernel = new Kernel('fn main() { }', {
                threads: () => [640, 320, 128] as [number, number, number]
            });
            const dispatch = resolveDispatch(kernel, {});
            // ceil(640/64)=10, ceil(320/1)=320, ceil(128/1)=128
            expect(dispatch).toEqual([10, 320, 128]);
        });
        it('pass-time threads override kernel threads', () => {
            const kernel = new Kernel('fn main() { }', { threads: 64 });
            const dispatch = resolveDispatch(kernel, {}, 128);
            // 128 / 64 = 2 dispatches
            expect(dispatch).toEqual([2, 1, 1]);
        });
        it('pass-time array threads are treated as total invocations', () => {
            const kernel = new Kernel('fn main() { }', { threads: 64 });
            const dispatch = resolveDispatch(kernel, {}, [192, 3, 3]);
            // ceil(192/64)=3, ceil(3/1)=3, ceil(3/1)=3
            expect(dispatch).toEqual([3, 3, 3]);
        });
        it('single-element pass-time thread tuples are treated as 1D total invocations', () => {
            const kernel = new Kernel('fn main() { }', { threads: 64 });
            const dispatch = resolveDispatch(kernel, {}, [128]);
            // [128] -> [128, 1, 1] -> 128 / 64 = 2 dispatches
            expect(dispatch).toEqual([2, 1, 1]);
        });
        it('auto-infers from single buffer binding', () => {
            const input = makeBuffer(
                Array.from({ length: 128 }, (_, i) => i),
                'f32',
                'r'
            );
            const kernel = new Kernel('fn main() { }');
            const dispatch = resolveDispatch(kernel, { input });
            // 128 / 64 = 2 dispatches
            expect(dispatch).toEqual([2, 1, 1]);
        });
        it('auto-infers from single input when output is excluded', () => {
            const input = makeBuffer(
                Array.from({ length: 64 }, (_, i) => i),
                'f32',
                'r'
            );
            const output = makeBuffer(
                Array.from({ length: 64 }, () => 0),
                'f32',
                'rw'
            );
            const kernel = new Kernel('fn main() { }', {
                outputs: { output: { definedBy: 'input' } }
            });
            const dispatch = resolveDispatch(kernel, { input, output });
            // 64 / 64 = 1 dispatch
            expect(dispatch).toEqual([1, 1, 1]);
        });
        it('auto-infers from single input when string outputs exclude the output binding', () => {
            const input = makeBuffer(
                Array.from({ length: 64 }, (_, i) => i),
                'f32',
                'r'
            );
            const output = makeBuffer(
                Array.from({ length: 64 }, () => 0),
                'f32',
                'rw'
            );
            const kernel = new Kernel('fn main() { }', {
                outputs: ['output']
            });
            const dispatch = resolveDispatch(kernel, { input, output });
            expect(dispatch).toEqual([1, 1, 1]);
        });
        it('throws when string threads references missing binding', () => {
            const kernel = new Kernel('fn main() { }', { threads: 'data' });
            expect(() =>
                resolveDispatch(kernel, { input: makeBuffer() })
            ).toThrow(/references input "data"/);
        });
        it('throws when cannot auto-infer (no buffers)', () => {
            const kernel = new Kernel('fn main() { }');
            expect(() => resolveDispatch(kernel, {})).toThrow(
                /no buffer bindings found/
            );
        });
        it('throws when cannot auto-infer (multiple ambiguous inputs)', () => {
            const a = makeBuffer([1, 2, 3], 'f32', 'r');
            const b = makeBuffer([4, 5, 6], 'f32', 'r');
            const kernel = new Kernel('fn main() { }');
            expect(() => resolveDispatch(kernel, { a, b })).toThrow(
                /2 input buffers found/
            );
        });
        it('handles ceil correctly for non-divisible sizes', () => {
            const kernel = new Kernel('fn main() { }', { threads: 100 });
            const dispatch = resolveDispatch(kernel, {});
            // ceil(100 / 64) = 2
            expect(dispatch).toEqual([2, 1, 1]);
        });
        // ----- Multi-dimensional workgroup edge cases (no thread explosion) -----
        it('scalar threads with 2D workgroup divides by x-axis only', () => {
            // workgroup [8, 8, 1]: scalar 1024 treated as [1024, 1, 1]
            // ceil(1024/8) = 128, ceil(1/8) = 1, ceil(1/1) = 1
            const kernel = new Kernel('fn main() { }', {
                threads: 1024,
                workgroupSize: [8, 8, 1]
            });
            const dispatch = resolveDispatch(kernel, {});
            expect(dispatch).toEqual([128, 1, 1]);
        });
        it('scalar threads with 3D workgroup divides by x-axis only', () => {
            // workgroup [4, 4, 4]: scalar 512 treated as [512, 1, 1]
            // ceil(512/4) = 128, ceil(1/4) = 1, ceil(1/4) = 1
            const kernel = new Kernel('fn main() { }', {
                threads: 512,
                workgroupSize: [4, 4, 4]
            });
            const dispatch = resolveDispatch(kernel, {});
            expect(dispatch).toEqual([128, 1, 1]);
        });
        it('array threads with 2D workgroup divides each dimension', () => {
            // threads [256, 256, 1] with workgroup [8, 8, 1]
            // → [ceil(256/8), ceil(256/8), ceil(1/1)] = [32, 32, 1]
            const kernel = new Kernel('fn main() { }', {
                workgroupSize: [8, 8, 1]
            });
            const dispatch = resolveDispatch(kernel, {}, [256, 256, 1]);
            expect(dispatch).toEqual([32, 32, 1]);
        });
        it('function threads with 2D workgroup divides each dimension', () => {
            const kernel = new Kernel('fn main() { }', {
                threads: () => [100, 100, 1] as [number, number, number],
                workgroupSize: [8, 8, 1]
            });
            const dispatch = resolveDispatch(kernel, {});
            // ceil(100/8)=13, ceil(100/8)=13, ceil(1/1)=1
            expect(dispatch).toEqual([13, 13, 1]);
        });
        it('pass-time array threads with non-default workgroup divides correctly', () => {
            const kernel = new Kernel('fn main() { }', {
                workgroupSize: [16, 16, 1]
            });
            const dispatch = resolveDispatch(kernel, {}, [64, 64, 1]);
            // ceil(64/16)=4, ceil(64/16)=4, ceil(1/1)=1
            expect(dispatch).toEqual([4, 4, 1]);
        });
        // ----- 2D thread array normalization -----
        it('2D pass-time array normalizes to 3D (z defaults to 1)', () => {
            const kernel = new Kernel('fn main() { }', {
                workgroupSize: [8, 8, 1]
            });
            const dispatch = resolveDispatch(kernel, {}, [64, 64]);
            // [64,64] → [64,64,1] → [ceil(64/8), ceil(64/8), ceil(1/1)] = [8, 8, 1]
            expect(dispatch).toEqual([8, 8, 1]);
        });
        it('2D function threads normalizes to 3D', () => {
            const kernel = new Kernel('fn main() { }', {
                threads: () => [128, 256] as [number, number],
                workgroupSize: [64, 1, 1]
            });
            const dispatch = resolveDispatch(kernel, {});
            // [128,256] → [128,256,1] → [ceil(128/64), ceil(256/1), ceil(1/1)] = [2, 256, 1]
            expect(dispatch).toEqual([2, 256, 1]);
        });
        it('function returning single number is treated as 1D total invocations', () => {
            const input = makeBuffer(
                Array.from({ length: 256 }, (_, i) => i),
                'f32',
                'r'
            );
            const kernel = new Kernel('fn main() { }', {
                threads: (data) => (data.input as Buffer).count
            });
            const dispatch = resolveDispatch(kernel, { input });
            // 256 / 64 = 4
            expect(dispatch).toEqual([4, 1, 1]);
        });
    });
});
// =============================================================================
// NODE CREATION TESTS
// =============================================================================
describe('Node Creation', () => {
    describe('validateOutputName()', () => {
        it('accepts valid names', () => {
            expect(() => validateOutputName('result')).not.toThrow();
            expect(() => validateOutputName('output')).not.toThrow();
            expect(() => validateOutputName('albedo')).not.toThrow();
        });
        it('rejects names starting with underscore', () => {
            expect(() => validateOutputName('_internal')).toThrow(
                /cannot start with "_"/
            );
        });
        it('rejects reserved names', () => {
            expect(() => validateOutputName('_id')).toThrow();
            expect(() => validateOutputName('_kernel')).toThrow();
        });
    });
    describe('createHandle()', () => {
        it('creates a handle with correct properties', () => {
            const fakeNode = { _id: Symbol('Test'), _label: 'TestNode' } as any;
            const handle = createHandle(fakeNode, 'result');
            expect(handle._name).toBe('result');
            expect(typeof handle._id).toBe('symbol');
            expect(handle._node).toBe(fakeNode);
            expect(handle._label).toBe('TestNode.result');
        });
    });
    describe('createNode()', () => {
        // We need mock GPU objects for createNode
        const mockPipeline = { label: 'mock' } as any as GPUComputePipeline;
        const mockLayout = { label: 'mock' } as any as GPUBindGroupLayout;
        it('creates a node with handles for all buffer-like bindings', () => {
            const kernel = new Kernel('fn main() { }');
            const result = makeBuffer([0, 0, 0], 'f32', 'rw');
            const debug = makeBuffer([0, 0, 0], 'f32', 'rw');
            const node = createNode({
                kernel,
                pipeline: mockPipeline,
                bindGroupLayout: mockLayout,
                bindingEntries: [],
                bounds: [3, 1, 1],
                dispatch: [1, 1, 1],
                bindings: { result, debug },
                shaderCode: 'fn main() {}',
                dependencies: []
            });
            // Handles should be directly on the node for ALL buffer bindings
            expect(node.result).toBeDefined();
            expect(node.result._name).toBe('result');
            expect(node.debug).toBeDefined();
            expect(node.debug._name).toBe('debug');
        });
        it('node has correct internal properties', () => {
            const kernel = new Kernel('fn main() { }');
            const node = createNode({
                kernel,
                pipeline: mockPipeline,
                bindGroupLayout: mockLayout,
                bindingEntries: [],
                bounds: [4, 1, 1],
                dispatch: [4, 1, 1],
                bindings: {},
                shaderCode: 'test',
                dependencies: []
            });
            expect(typeof node._id).toBe('symbol');
            expect(node._label).toBe('Node');
            expect(node._kernel).toBe(kernel);
            expect(node._pipeline).toBe(mockPipeline);
            expect(node._bindGroupLayout).toBe(mockLayout);
            expect(node._bounds).toEqual([4, 1, 1]);
            expect(node._dispatch).toEqual([4, 1, 1]);
            expect(node._dependencies).toEqual([]);
        });
        it('supports explicit node labels', () => {
            const kernel = new Kernel('fn main() { }');
            const node = createNode({
                kernel,
                pipeline: mockPipeline,
                bindGroupLayout: mockLayout,
                bindingEntries: [],
                bounds: [1, 1, 1],
                dispatch: [1, 1, 1],
                bindings: {},
                shaderCode: 'test',
                dependencies: [],
                label: 'blur-pass'
            });
            expect(node._label).toBe('blur-pass');
        });
        it('tracks dependencies from Handle inputs', () => {
            const kernel = new Kernel('fn main() { }');
            const parentNode = createNode({
                kernel,
                pipeline: mockPipeline,
                bindGroupLayout: mockLayout,
                bindingEntries: [],
                bounds: [1, 1, 1],
                dispatch: [1, 1, 1],
                bindings: {},
                shaderCode: 'test',
                dependencies: []
            });
            const childNode = createNode({
                kernel,
                pipeline: mockPipeline,
                bindGroupLayout: mockLayout,
                bindingEntries: [],
                bounds: [1, 1, 1],
                dispatch: [1, 1, 1],
                bindings: {},
                shaderCode: 'test',
                dependencies: [parentNode]
            });
            expect(childNode._dependencies).toHaveLength(1);
            expect(childNode._dependencies[0]).toBe(parentNode);
        });
        it('does not create handles for non-buffer bindings', () => {
            const kernel = new Kernel('fn main() { }');
            const node = createNode({
                kernel,
                pipeline: mockPipeline,
                bindGroupLayout: mockLayout,
                bindingEntries: [],
                bounds: [1, 1, 1],
                dispatch: [1, 1, 1],
                bindings: {},
                shaderCode: 'test',
                dependencies: []
            });
            const handles = getNodeHandles(node);
            expect(Object.keys(handles)).toEqual([]);
        });
        it('does not create handles for Uniform bindings', () => {
            const kernel = new Kernel('fn main() { }');
            const uniform = makeUniform(2.0, 'f32');
            const node = createNode({
                kernel,
                pipeline: mockPipeline,
                bindGroupLayout: mockLayout,
                bindingEntries: [],
                bounds: [1, 1, 1],
                dispatch: [1, 1, 1],
                bindings: { multiplier: uniform },
                shaderCode: 'test',
                dependencies: []
            });
            const handles = getNodeHandles(node);
            expect(Object.keys(handles)).toEqual([]);
            expect(isBufferLike(uniform)).toBe(false);
        });
    });
});
// =============================================================================
// PIPELINE CACHE TESTS
// =============================================================================
describe('Pipeline Cache', () => {
    // Minimal mock GPUDevice for pipeline cache testing
    function createMockDevice(): GPUDevice {
        const mockBindGroupLayout = { label: 'mock-layout' };
        const mockPipeline = {
            label: 'mock-pipeline',
            getBindGroupLayout: vi.fn().mockReturnValue(mockBindGroupLayout)
        };
        return {
            createShaderModule: vi
                .fn()
                .mockReturnValue({ label: 'mock-module' }),
            createComputePipeline: vi.fn().mockReturnValue(mockPipeline)
        } as any;
    }
    it('creates a pipeline on first call', () => {
        const cache = new PipelineCache();
        const device = createMockDevice();
        const result = cache.getOrCreate(device, 'fn main() { }');
        expect(result.pipeline).toBeDefined();
        expect(result.bindGroupLayout).toBeDefined();
        expect(cache.size).toBe(1);
    });
    it('returns cached pipeline for same shader code', () => {
        const cache = new PipelineCache();
        const device = createMockDevice();
        const first = cache.getOrCreate(device, 'fn main() { }');
        const second = cache.getOrCreate(device, 'fn main() { }');
        expect(first).toBe(second); // Same object reference
        expect(cache.size).toBe(1);
        // createShaderModule should only be called once
        expect(device.createShaderModule).toHaveBeenCalledTimes(1);
    });
    it('creates new pipeline for different shader code', () => {
        const cache = new PipelineCache();
        const device = createMockDevice();
        cache.getOrCreate(device, 'fn main() { /* A */ }');
        cache.getOrCreate(device, 'fn main() { /* B */ }');
        expect(cache.size).toBe(2);
        expect(device.createShaderModule).toHaveBeenCalledTimes(2);
    });
    it('clears all cached pipelines', () => {
        const cache = new PipelineCache();
        const device = createMockDevice();
        cache.getOrCreate(device, 'fn main() { }');
        expect(cache.size).toBe(1);
        cache.clear();
        expect(cache.size).toBe(0);
    });
});
// =============================================================================
// CALLABLE OPERATION MATERIALIZATION TESTS
// =============================================================================
describe('Callable operation materialization', () => {
    function createMockVoltenContext(): VoltenContext {
        const mockBindGroupLayout = { label: 'mock-layout' };
        const mockPipeline = {
            label: 'mock-pipeline',
            getBindGroupLayout: vi.fn().mockReturnValue(mockBindGroupLayout)
        };
        const mockDevice = {
            createShaderModule: vi
                .fn()
                .mockReturnValue({ label: 'mock-module' }),
            createComputePipeline: vi.fn().mockReturnValue(mockPipeline)
        } as any;
        return new VoltenContext(mockDevice);
    }
    it('throws when standard uniform mode is requested without extension support', () => {
        expect(
            () =>
                new VoltenContext({} as GPUDevice, {
                    uniformLayoutMode: 'standard'
                })
        ).toThrow(/requires WGSL extension "uniform_buffer_standard_layout"/);
    });
    it('checks WGSL language features without detaching the has() method', () => {
        const originalNavigatorDescriptor = Object.getOwnPropertyDescriptor(
            globalThis,
            'navigator'
        );
        Object.defineProperty(globalThis, 'navigator', {
            configurable: true,
            value: {
                gpu: {
                    wgslLanguageFeatures: {
                        featureName: 'uniform_buffer_standard_layout',
                        has(
                            this: {
                                featureName: string;
                            },
                            name: string
                        ) {
                            return name === this.featureName;
                        }
                    }
                }
            }
        });
        try {
            expect(
                supportsWgslLanguageFeature('uniform_buffer_standard_layout')
            ).toBe(true);
        } finally {
            if (originalNavigatorDescriptor) {
                Object.defineProperty(
                    globalThis,
                    'navigator',
                    originalNavigatorDescriptor
                );
            } else {
                delete (
                    globalThis as {
                        navigator?: unknown;
                    }
                ).navigator;
            }
        }
    });
    it('auto mode uses standard layout when extension is available', () => {
        const originalNavigatorDescriptor = Object.getOwnPropertyDescriptor(
            globalThis,
            'navigator'
        );
        Object.defineProperty(globalThis, 'navigator', {
            configurable: true,
            value: {
                gpu: {
                    wgslLanguageFeatures: {
                        has: (name: string) =>
                            name === 'uniform_buffer_standard_layout'
                    }
                }
            }
        });
        try {
            const v = createMockVoltenContext();
            const S = struct('AutoS', { x: 'f32' });
            const Params = struct('AutoParams', {
                a: S,
                b: 'f32'
            });
            const params = makeUniform({ a: { x: 1 }, b: 2 }, Params);
            const kernel = new Kernel('fn main() { }', { threads: 1 });
            const node = invoke(kernel, { params });
            const dispatch = getDispatch(v, node);
            expect(dispatch._shaderCode).toContain(
                'requires uniform_buffer_standard_layout;'
            );
            expect(dispatch._shaderCode).not.toContain('@align(16) b: f32,');
            expect(params.byteLength).toBe(8);
        } finally {
            if (originalNavigatorDescriptor) {
                Object.defineProperty(
                    globalThis,
                    'navigator',
                    originalNavigatorDescriptor
                );
            } else {
                delete (
                    globalThis as {
                        navigator?: unknown;
                    }
                ).navigator;
            }
        }
    });
    it('creates a node from kernel + buffer bindings', () => {
        const v = createMockVoltenContext();
        const input = makeBuffer([1, 2, 3, 4], 'f32', 'r');
        const output = makeBuffer([0, 0, 0, 0], 'f32', 'rw');
        const kernel = new Kernel(
            `
fn main(gid: vec3u) {
    output[gid.x] = input[gid.x] * 2.0;
}
`,
            { outputs: { output: { definedBy: 'input' } }, threads: 'input' }
        );
        const node = invoke(kernel, { input, output });
        const dispatch = getDispatch(v, node);
        expect(dispatch._kernel.source).toBe(kernel.source);
        expect(dispatch._pipeline).toBeDefined();
        expect(dispatch._dispatch).toEqual([1, 1, 1]); // 4 elements / 64 workgroup
        expect(dispatch._shaderCode).toContain('@group(0) @binding(0)');
        expect(dispatch._shaderCode).toContain('@group(0) @binding(1)');
        expect(dispatch._shaderCode).toContain('@compute');
    });
    it('assigns generated labels to kernels and operation nodes', () => {
        const input = makeBuffer([1, 2, 3, 4], 'f32', 'r');
        const kernel = new Kernel('fn main(gid: vec3u) { }');
        const node = invoke(kernel, { input });
        expect(kernel.label).toMatch(/^Kernel#/);
        expect(node._label.startsWith(`${kernel.label}::node#`)).toBe(true);
    });
    it('respects explicit labels for buffers, kernels, uniforms, and nodes', () => {
        const input = new Buffer([1, 2, 3, 4], 'f32', 'r', {
            label: 'input-buffer'
        });
        const output = new Buffer([0, 0, 0, 0], 'f32', 'rw', {
            label: 'output-buffer'
        });
        const multiplier = new Uniform(2.0, 'f32', {
            label: 'multiplier-uniform'
        });
        const kernel = new Kernel('fn main(gid: vec3u) { }', {
            label: 'scale-kernel',
            threads: 'input'
        });
        const node = invoke(
            kernel,
            { input, output, multiplier },
            { label: 'scale-node' }
        );
        expect(input.label).toBe('input-buffer');
        expect(output.label).toBe('output-buffer');
        expect(multiplier.label).toBe('multiplier-uniform');
        expect(kernel.label).toBe('scale-kernel');
        expect(node._label).toBe('scale-node');
        expect(node.output._label).toBe('scale-node.output');
    });
    it('output handles are accessible directly on the node', () => {
        const input = makeBuffer([1, 2, 3, 4], 'f32', 'r');
        const output = makeBuffer([0, 0, 0, 0], 'f32', 'rw');
        const kernel = new Kernel('fn main(gid: vec3u) { }', {
            outputs: { output: { definedBy: 'input' } },
            threads: 4
        });
        const node = invoke(kernel, { input, output });
        // Handles are directly on the node for ALL buffer bindings
        expect(node.output).toBeDefined();
        expect(node.output._name).toBe('output');
        expect(node.output._node).toBe(node);
        // Input buffer also gets a handle
        expect(node.input).toBeDefined();
        expect(node.input._name).toBe('input');
    });
    it('chains passes via Handle dependencies', () => {
        const v = createMockVoltenContext();
        const source = makeBuffer([1, 2, 3, 4], 'f32', 'r');
        const mid = makeBuffer([0, 0, 0, 0], 'f32', 'rw');
        const result = makeBuffer([0, 0, 0, 0], 'f32', 'rw');
        const K1 = new Kernel('fn main(gid: vec3u) { }', {
            outputs: { mid: { definedBy: 'source' } },
            threads: 'source'
        });
        const K2 = new Kernel('fn main(gid: vec3u) { }', {
            outputs: { result: { definedBy: 'data' } },
            threads: 4
        });
        const A = invoke(K1, { source, mid });
        const B = invoke(K2, { data: A.mid, result });
        const dispatchA = getDispatch(v, A);
        const dispatchB = getDispatch(v, B);
        // B should depend on A
        expect(dispatchB._dependencies).toHaveLength(1);
        expect(dispatchB._dependencies[0]._id).toBe(dispatchA._id);
    });
    it('caches pipelines for identical shaders', () => {
        const v = createMockVoltenContext();
        const input = makeBuffer([1, 2], 'f32', 'r');
        const kernel = new Kernel('fn main(gid: vec3u) { }', {
            threads: 2
        });
        const A = invoke(kernel, { input });
        const B = invoke(kernel, { input });
        const dispatchA = getDispatch(v, A);
        const dispatchB = getDispatch(v, B);
        // Same pipeline since same kernel + same binding types
        expect(dispatchA._pipeline).toBe(dispatchB._pipeline);
        expect(v._pipelineCache.size).toBe(1);
    });
    it('pass-time threads override kernel threads', () => {
        const v = createMockVoltenContext();
        const input = makeBuffer([1, 2, 3, 4], 'f32', 'r');
        const kernel = new Kernel('fn main(gid: vec3u) { }', {
            threads: 4
        });
        const node = invoke(kernel, { input }, { threads: 256 });
        const dispatch = getDispatch(v, node);
        // 256 / 64 = 4
        expect(dispatch._dispatch).toEqual([4, 1, 1]);
    });
    it('accepts a single-element pass-time thread tuple', () => {
        const v = createMockVoltenContext();
        const input = makeBuffer([1, 2, 3, 4], 'f32', 'r');
        const kernel = new Kernel('fn main(gid: vec3u) { }', {
            threads: 4
        });
        const node = invoke(kernel, { input }, { threads: [256] });
        const dispatch = getDispatch(v, node);
        expect(dispatch._bounds).toEqual([256, 1, 1]);
        expect(dispatch._dispatch).toEqual([4, 1, 1]);
    });
    it('removes the context-bound pass constructor', () => {
        const v = createMockVoltenContext();
        expect((v as any).pass).toBeUndefined();
    });
    it('includes binding WGSL in assembled shader code', () => {
        const v = createMockVoltenContext();
        const input = makeBuffer([1, 2], 'f32', 'r');
        const output = makeBuffer([0, 0], 'f32', 'rw');
        const kernel = new Kernel(
            `
fn main(gid: vec3u) {
    output[gid.x] = input[gid.x];
}
`,
            { outputs: { output: { definedBy: 'input' } }, threads: 2 }
        );
        const node = invoke(kernel, { input, output });
        const dispatch = getDispatch(v, node);
        // Shader should contain binding declarations
        expect(dispatch._shaderCode).toContain(
            'var<storage, read> input: array<f32>'
        );
        expect(dispatch._shaderCode).toContain(
            'var<storage, read_write> output: array<f32>'
        );
        // AND kernel logic
        expect(dispatch._shaderCode).toContain(
            '@builtin(global_invocation_id)'
        );
        expect(dispatch._shaderCode).toContain(
            '@compute @workgroup_size(64, 1, 1)'
        );
    });
    it('supports Uniform bindings', () => {
        const v = createMockVoltenContext();
        const input = makeBuffer([1, 2, 3, 4], 'f32', 'r');
        const output = makeBuffer([0, 0, 0, 0], 'f32', 'rw');
        const multiplier = makeUniform(2.0, 'f32');
        const kernel = new Kernel(
            `
fn main(gid: vec3u) {
    output[gid.x] = input[gid.x] * multiplier;
}
`,
            { threads: 'input' }
        );
        const node = invoke(kernel, { input, output, multiplier });
        const dispatch = getDispatch(v, node);
        expect(dispatch._bindingEntries).toHaveLength(4);
        const uniformEntry = dispatch._bindingEntries.find(
            (e) => e.name === 'multiplier'
        );
        const boundsEntry = dispatch._bindingEntries.find(
            (e) => e.name === VOLTEN_BOUNDS_NAME
        );
        expect(uniformEntry).toBeDefined();
        expect(boundsEntry).toBeDefined();
        expect(boundsEntry!.wgslAddressSpace).toBe('uniform');
        expect(boundsEntry!.wgslType).toBe('vec4u');
        expect(uniformEntry!.wgslAddressSpace).toBe('uniform');
        expect(uniformEntry!.wgslType).toBe('f32');
        expect(dispatch._shaderCode).toContain('var<uniform> multiplier: f32;');
        expect(dispatch._shaderCode).toContain(
            `var<uniform> ${VOLTEN_BOUNDS_NAME}: vec4u;`
        );
        expect((node as any).multiplier).toBeUndefined();
    });
    it('injects hidden bounds and stores logical bounds by default', () => {
        const v = createMockVoltenContext();
        const input = makeBuffer([1, 2, 3, 4], 'f32', 'rw');
        const kernel = new Kernel(
            'fn main(gid: vec3u) { data[gid.x] = data[gid.x]; }',
            {
                threads: 4
            }
        );
        const node = invoke(kernel, { data: input });
        const dispatch = getDispatch(v, node);
        expect(dispatch._bounds).toEqual([4, 1, 1]);
        expect(dispatch._dispatch).toEqual([1, 1, 1]);
        expect(dispatch._shaderCode).toContain(VOLTEN_BOUNDS_NAME);
        expect(dispatch._shaderCode).toContain(
            '_volten_user_main_entrypoint_wrapper'
        );
    });
    it('allows opting out of hidden bounds with unsafeManualBounds', () => {
        const v = createMockVoltenContext();
        const input = makeBuffer([1, 2, 3, 4], 'f32', 'rw');
        const kernel = new Kernel(
            'fn main(gid: vec3u) { data[gid.x] = data[gid.x]; }',
            {
                threads: 4,
                unsafeManualBounds: true
            }
        );
        const node = invoke(kernel, { data: input });
        const dispatch = getDispatch(v, node);
        expect(
            dispatch._bindingEntries.find((e) => e.name === VOLTEN_BOUNDS_NAME)
        ).toBeUndefined();
        expect(dispatch._shaderCode).not.toContain(VOLTEN_BOUNDS_NAME);
        expect(dispatch._shaderCode).not.toContain(
            '_volten_user_main_entrypoint_wrapper'
        );
    });
    it('throws when a barrier kernel would require a guarded partial workgroup', () => {
        const v = createMockVoltenContext();
        const input = makeBuffer([1, 2, 3, 4], 'f32', 'rw');
        const kernel = new Kernel(
            `
fn main(gid: vec3u) {
    workgroupBarrier();
    data[gid.x] = data[gid.x];
}
`,
            { threads: 4 }
        );
        const node = invoke(kernel, { data: input });
        expect(() => getDispatch(v, node)).toThrow(/unsafeManualBounds: true/);
    });
    it('handles kernel with no outputs (side-effect only)', () => {
        const v = createMockVoltenContext();
        const data = makeBuffer([1, 2, 3], 'f32', 'rw');
        const kernel = new Kernel('fn main(gid: vec3u) { }', {
            threads: 3
        });
        const node = invoke(kernel, { data });
        const dispatch = getDispatch(v, node);
        const handles = getNodeHandles(dispatch);
        expect(Object.keys(handles)).toEqual(['data']);
        const outputs = getNodeOutputHandles(dispatch);
        expect(Object.keys(outputs)).toEqual([]);
    });
    it('handles multiple outputs', () => {
        const v = createMockVoltenContext();
        const input = makeBuffer([1, 2, 3], 'f32', 'r');
        const albedo = makeBuffer([0, 0, 0], 'f32', 'rw');
        const normal = makeBuffer([0, 0, 0], 'f32', 'rw');
        const kernel = new Kernel('fn main(gid: vec3u) { }', {
            outputs: {
                albedo: { definedBy: 'input' },
                normal: { definedBy: 'input' }
            },
            threads: 'input'
        });
        const node = invoke(kernel, { input, albedo, normal });
        expect(node.albedo._name).toBe('albedo');
        expect(node.normal._name).toBe('normal');
        const dispatch = getDispatch(v, node);
        const handles = getNodeHandles(dispatch);
        // All buffer bindings get handles (input too)
        expect(Object.keys(handles).sort()).toEqual([
            'albedo',
            'input',
            'normal'
        ]);
        const outputs = getNodeOutputHandles(dispatch);
        expect(Object.keys(outputs).sort()).toEqual(['albedo', 'normal']);
    });
    it('creates handles for in-place buffer bindings without output declarations', () => {
        const data = makeBuffer([1, 2, 3, 4], 'f32', 'rw');
        // No outputs declared
        const kernel = new Kernel('fn main(gid: vec3u) { }', {
            threads: 4
        });
        const A = invoke(kernel, { data });
        // Handle should still be available for the buffer binding
        expect(A.data).toBeDefined();
        expect(A.data._name).toBe('data');
        expect(A.data._node).toBe(A);
    });
    it('enables in-place buffer chaining without output declarations', () => {
        const v = createMockVoltenContext();
        const buf = makeBuffer([1, 2, 3, 4], 'f32', 'rw');
        const kernel = new Kernel('fn main(gid: vec3u) { }', {
            threads: 4
        });
        // Chain: A → B → C all modifying the same buffer in-place
        const A = invoke(kernel, { inout: buf });
        const B = invoke(kernel, { inout: A.inout });
        const C = invoke(kernel, { inout: B.inout });
        const dispatchA = getDispatch(v, A);
        const dispatchB = getDispatch(v, B);
        const dispatchC = getDispatch(v, C);
        // C should depend on B, which depends on A
        expect(dispatchC._dependencies).toHaveLength(1);
        expect(dispatchC._dependencies[0]._id).toBe(dispatchB._id);
        expect(dispatchB._dependencies).toHaveLength(1);
        expect(dispatchB._dependencies[0]._id).toBe(dispatchA._id);
    });
    it('Handle type resolves from source buffer (not hardcoded)', () => {
        const v = createMockVoltenContext();
        const positions = new Buffer([[1, 2, 3]], 'vec3f' as any, 'rw');
        const kernel = new Kernel('fn main(gid: vec3u) { }', { threads: 1 });
        const A = invoke(kernel, { positions });
        const B = invoke(kernel, { data: A.positions });
        const dispatchB = getDispatch(v, B);
        // The binding entry for 'data' should have vec3f type from the source
        const dataEntry = dispatchB._bindingEntries.find(
            (e) => e.name === 'data'
        );
        expect(dataEntry).toBeDefined();
        expect(dataEntry!.wgslType).toBe('array<vec3f>');
        expect(dataEntry!.wgslAccess).toBe('read_write');
    });
});

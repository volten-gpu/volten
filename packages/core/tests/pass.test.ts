/**
 * Tests for v.pass() — binding generation, shader assembly,
 * dispatch resolution, and Node creation.
 *
 * Since we're in a Node.js environment without a real GPU, tests cover:
 * 1. Binding classification (Buffer, RawBuffer, Handle)
 * 2. WGSL binding declaration generation
 * 3. Full shader assembly
 * 4. Thread dispatch resolution
 * 5. Error handling & validation
 * 6. v.pass() integration with mocked GPUDevice
 */
import { describe, it, expect, vi } from 'vitest';
import {
    generateBindings,
    generateBindingWgsl,
    assembleFullShader,
    resolveBounds,
    resolveDispatch
} from '../src/kernel/bindings.js';
import { Kernel } from '../src/kernel/kernel.js';
import { VOLTEN_INTERNAL_BOUNDS_NAME } from '../src/kernel/builtins.js';
import { Buffer } from '../src/data/buffer.js';
import { RawBuffer } from '../src/data/raw-buffer.js';
import { Uniform } from '../src/data/uniform.js';
import { array, struct } from '../src/types/schema.js';
import {
    createNode,
    createHandle,
    validateOutputName,
    getNodeOutputs,
    isBufferLike,
    type Handle,
    type Node
} from '../src/graph/node.js';
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
        _bindings: { [name]: backingBuffer }
    } as any;
    return {
        _id: Symbol(`Handle:${name}`),
        _node: fakeNode,
        _name: name
    };
}

// =============================================================================
// BINDING GENERATION TESTS
// =============================================================================

describe('Binding Generation', () => {
    describe('generateBindings()', () => {
        it('classifies a read-only Buffer binding', () => {
            const kernel = new Kernel('fn main() { }');
            const buf = makeBuffer([1, 2, 3], 'f32', 'r');
            const entries = generateBindings({ input: buf }, kernel);

            expect(entries).toHaveLength(1);
            expect(entries[0].index).toBe(0);
            expect(entries[0].name).toBe('input');
            expect(entries[0].wgslType).toBe('array<f32>');
            expect(entries[0].wgslAccess).toBe('read');
            expect(entries[0].isHandle).toBe(false);
        });

        it('classifies a read-write Buffer binding', () => {
            const kernel = new Kernel('fn main() { }');
            const buf = makeBuffer([1, 2, 3], 'f32', 'rw');
            const entries = generateBindings({ output: buf }, kernel);

            expect(entries[0].wgslAccess).toBe('read_write');
        });

        it('classifies a RawBuffer binding', () => {
            const kernel = new Kernel('fn main() { }');
            const raw = makeRawBuffer(64, 'array<vec4f>');
            const entries = generateBindings({ data: raw }, kernel);

            expect(entries).toHaveLength(1);
            expect(entries[0].name).toBe('data');
            expect(entries[0].wgslType).toBe('array<vec4f>');
            expect(entries[0].isHandle).toBe(false);
        });

        it('classifies a Uniform binding', () => {
            const kernel = new Kernel('fn main() { }');
            const uniform = makeUniform([1, 2, 3, 4], 'vec4f');
            const entries = generateBindings({ transform: uniform }, kernel);

            expect(entries).toHaveLength(1);
            expect(entries[0].name).toBe('transform');
            expect(entries[0].wgslType).toBe('vec4f');
            expect(entries[0].wgslAddressSpace).toBe('uniform');
            expect(entries[0].wgslAccess).toBeUndefined();
            expect(entries[0].isHandle).toBe(false);
        });

        it('classifies a Handle binding with resolved type from source', () => {
            const kernel = new Kernel('fn main() { }');
            const handle = makeMockHandle('prevOutput');
            const entries = generateBindings({ prevData: handle }, kernel);

            expect(entries).toHaveLength(1);
            expect(entries[0].name).toBe('prevData');
            // Resolved from the backing buffer (f32, read-only)
            expect(entries[0].wgslType).toBe('array<f32>');
            expect(entries[0].wgslAccess).toBe('read');
            expect(entries[0].isHandle).toBe(true);
        });

        it('assigns sequential binding indices', () => {
            const kernel = new Kernel('fn main() { }');
            const a = makeBuffer([1], 'f32', 'r');
            const b = makeBuffer([2], 'f32', 'rw');
            const c = makeRawBuffer(4, 'array<u32>');

            const entries = generateBindings({ a, b, c }, kernel);

            expect(entries[0].index).toBe(0);
            expect(entries[1].index).toBe(1);
            expect(entries[2].index).toBe(2);
        });

        it('does not throw when kernel outputs are not provided (outputs are optional hints)', () => {
            const kernel = new Kernel('fn main() { }', {
                outputs: { result: { definedBy: 'input' } }
            });
            const input = makeBuffer([1, 2, 3], 'f32', 'r');

            // Should NOT throw — outputs are purely for future pool allocation
            const entries = generateBindings({ input }, kernel);
            expect(entries).toHaveLength(1);
        });

        it('accepts when kernel output is provided along with its declaration', () => {
            const kernel = new Kernel('fn main() { }', {
                outputs: { result: { definedBy: 'input' } }
            });
            const input = makeBuffer([1, 2, 3], 'f32', 'r');
            const result = makeBuffer([0, 0, 0], 'f32', 'rw');

            const entries = generateBindings({ input, result }, kernel);
            expect(entries).toHaveLength(2);
        });

        it('throws on number binding with helpful hint', () => {
            const kernel = new Kernel('fn main() { }');

            expect(() =>
                generateBindings({ scale: 2.5 as any }, kernel)
            ).toThrow(/Raw scalar bindings are not supported/);
            expect(() =>
                generateBindings({ scale: 2.5 as any }, kernel)
            ).toThrow(/new Buffer/);
        });

        it('throws on unsupported binding type', () => {
            const kernel = new Kernel('fn main() { }');

            expect(() =>
                generateBindings({ name: 'hello' as any }, kernel)
            ).toThrow(/unsupported type/);
        });

        it('handles empty bindings with no kernel outputs', () => {
            const kernel = new Kernel('fn main() { }');
            const entries = generateBindings({}, kernel);
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
            const kernel = new Kernel('fn main() { }');
            const buf = makeBuffer([1], 'f32', 'r');
            const entries = generateBindings({ input: buf }, kernel);
            const wgsl = generateBindingWgsl(entries);

            expect(wgsl).toBe(
                '@group(0) @binding(0) var<storage, read> input: array<f32>;'
            );
        });

        it('generates correct binding declaration for read-write buffer', () => {
            const kernel = new Kernel('fn main() { }');
            const buf = makeBuffer([1], 'f32', 'rw');
            const entries = generateBindings({ output: buf }, kernel);
            const wgsl = generateBindingWgsl(entries);

            expect(wgsl).toBe(
                '@group(0) @binding(0) var<storage, read_write> output: array<f32>;'
            );
        });

        it('generates multiple binding declarations', () => {
            const kernel = new Kernel('fn main() { }');
            const input = makeBuffer([1], 'f32', 'r');
            const output = makeBuffer([1], 'f32', 'rw');
            const entries = generateBindings({ input, output }, kernel);
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
            const kernel = new Kernel('fn main() { }');
            const scale = makeUniform(2.0, 'f32');
            const entries = generateBindings({ scale }, kernel);
            const wgsl = generateBindingWgsl(entries);

            expect(wgsl).toBe('@group(0) @binding(0) var<uniform> scale: f32;');
        });

        it('handles vec3f buffer type', () => {
            const kernel = new Kernel('fn main() { }');
            const buf = new Buffer([[1, 2, 3]], 'vec3f' as any, 'r');
            const entries = generateBindings({ positions: buf }, kernel);
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
            const entries = generateBindings({ input, output }, kernel);
            const shader = assembleFullShader(kernel, entries);

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
            const entries = generateBindings({}, kernel);
            const shader = assembleFullShader(kernel, entries);

            expect(shader).toContain('@compute @workgroup_size(128, 1, 1)');
        });

        it('includes builtin expansion from kernel assembly', () => {
            const kernel = new Kernel('fn main(gid: vec3u) { }');
            const entries = generateBindings({}, kernel);
            const shader = assembleFullShader(kernel, entries);

            expect(shader).toContain(
                '@builtin(global_invocation_id) gid: vec3u'
            );
        });

        it('returns kernel source alone when no bindings', () => {
            const kernel = new Kernel('fn main() { }');
            const entries = generateBindings({}, kernel);
            const shader = assembleFullShader(kernel, entries);

            expect(shader).toBe(kernel.assembledSource);
        });

        it('auto-emits struct declarations for struct-typed storage bindings', () => {
            const kernel = new Kernel('fn main() { }');
            const Particle = struct('Particle', {
                mass: 'f32'
            });
            const particles = new Buffer([{ mass: 1 }], Particle, 'r');
            const entries = generateBindings({ particles }, kernel);
            const shader = assembleFullShader(kernel, entries);

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
            const entries = generateBindings({ params }, kernel, {
                uniformLayoutMode: 'classic'
            });
            const shader = assembleFullShader(kernel, entries, {
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
            const entries = generateBindings({ params }, kernel, {
                uniformLayoutMode: 'standard'
            });
            const shader = assembleFullShader(kernel, entries, {
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
            const entries = generateBindings({ params }, kernel, {
                uniformLayoutMode: 'classic'
            });

            expect(() =>
                assembleFullShader(kernel, entries, {
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
                kernel,
                { uniformLayoutMode: 'classic' }
            );

            expect(() =>
                assembleFullShader(kernel, entries, {
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
                threads: (_data) => [640, 320, 128] as [number, number, number]
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
            const fakeNode = { _id: Symbol('Test') } as any;
            const handle = createHandle(fakeNode, 'result');

            expect(handle._name).toBe('result');
            expect(typeof handle._id).toBe('symbol');
            expect(handle._node).toBe(fakeNode);
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
            expect(node._kernel).toBe(kernel);
            expect(node._pipeline).toBe(mockPipeline);
            expect(node._bindGroupLayout).toBe(mockLayout);
            expect(node._bounds).toEqual([4, 1, 1]);
            expect(node._dispatch).toEqual([4, 1, 1]);
            expect(node._dependencies).toEqual([]);
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

            const outputs = getNodeOutputs(node);
            expect(Object.keys(outputs)).toEqual([]);
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

            const outputs = getNodeOutputs(node);
            expect(Object.keys(outputs)).toEqual([]);
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
// v.pass() INTEGRATION TESTS
// =============================================================================

describe('VoltenContext.pass()', () => {
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
                        has(this: { featureName: string }, name: string) {
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
                delete (globalThis as { navigator?: unknown }).navigator;
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
            const node = v.pass(kernel, { params });

            expect(node._shaderCode).toContain(
                'requires uniform_buffer_standard_layout;'
            );
            expect(node._shaderCode).not.toContain('@align(16) b: f32,');
            expect(params.byteLength).toBe(8);
        } finally {
            if (originalNavigatorDescriptor) {
                Object.defineProperty(
                    globalThis,
                    'navigator',
                    originalNavigatorDescriptor
                );
            } else {
                delete (globalThis as { navigator?: unknown }).navigator;
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

        const node = v.pass(kernel, { input, output });

        expect(node._kernel).toBe(kernel);
        expect(node._pipeline).toBeDefined();
        expect(node._dispatch).toEqual([1, 1, 1]); // 4 elements / 64 workgroup
        expect(node._shaderCode).toContain('@group(0) @binding(0)');
        expect(node._shaderCode).toContain('@group(0) @binding(1)');
        expect(node._shaderCode).toContain('@compute');
    });

    it('output handles are accessible directly on the node', () => {
        const v = createMockVoltenContext();
        const input = makeBuffer([1, 2, 3, 4], 'f32', 'r');
        const output = makeBuffer([0, 0, 0, 0], 'f32', 'rw');

        const kernel = new Kernel('fn main(gid: vec3u) { }', {
            outputs: { output: { definedBy: 'input' } },
            threads: 4
        });

        const node = v.pass(kernel, { input, output });

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

        const A = v.pass(K1, { source, mid });
        const B = v.pass(K2, { data: A.mid, result });

        // B should depend on A
        expect(B._dependencies).toHaveLength(1);
        expect(B._dependencies[0]._id).toBe(A._id);
    });

    it('caches pipelines for identical shaders', () => {
        const v = createMockVoltenContext();
        const input = makeBuffer([1, 2], 'f32', 'r');

        const kernel = new Kernel('fn main(gid: vec3u) { }', {
            threads: 2
        });

        const A = v.pass(kernel, { input });
        const B = v.pass(kernel, { input });

        // Same pipeline since same kernel + same binding types
        expect(A._pipeline).toBe(B._pipeline);
        expect(v._pipelineCache.size).toBe(1);
    });

    it('pass-time threads override kernel threads', () => {
        const v = createMockVoltenContext();
        const input = makeBuffer([1, 2, 3, 4], 'f32', 'r');

        const kernel = new Kernel('fn main(gid: vec3u) { }', {
            threads: 4
        });

        const node = v.pass(kernel, { input }, { threads: 256 });

        // 256 / 64 = 4
        expect(node._dispatch).toEqual([4, 1, 1]);
    });

    it('accepts a single-element pass-time thread tuple', () => {
        const v = createMockVoltenContext();
        const input = makeBuffer([1, 2, 3, 4], 'f32', 'r');

        const kernel = new Kernel('fn main(gid: vec3u) { }', {
            threads: 4
        });

        const node = v.pass(kernel, { input }, { threads: [256] });

        expect(node._bounds).toEqual([256, 1, 1]);
        expect(node._dispatch).toEqual([4, 1, 1]);
    });

    it('throws on non-Kernel first argument', () => {
        const v = createMockVoltenContext();

        expect(() => v.pass('not a kernel' as any, {})).toThrow(
            /must be a Kernel instance/
        );
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

        const node = v.pass(kernel, { input, output });

        // Shader should contain binding declarations
        expect(node._shaderCode).toContain(
            'var<storage, read> input: array<f32>'
        );
        expect(node._shaderCode).toContain(
            'var<storage, read_write> output: array<f32>'
        );
        // AND kernel logic
        expect(node._shaderCode).toContain('@builtin(global_invocation_id)');
        expect(node._shaderCode).toContain(
            '@compute @workgroup_size(64, 1, 1)'
        );
    });

    it('supports Uniform bindings in v.pass()', () => {
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

        const node = v.pass(kernel, { input, output, multiplier });

        expect(node._bindingEntries).toHaveLength(4);
        const uniformEntry = node._bindingEntries.find(
            (e) => e.name === 'multiplier'
        );
        const boundsEntry = node._bindingEntries.find(
            (e) => e.name === VOLTEN_INTERNAL_BOUNDS_NAME
        );
        expect(uniformEntry).toBeDefined();
        expect(boundsEntry).toBeDefined();
        expect(boundsEntry!.wgslAddressSpace).toBe('uniform');
        expect(boundsEntry!.wgslType).toBe('vec4u');
        expect(uniformEntry!.wgslAddressSpace).toBe('uniform');
        expect(uniformEntry!.wgslType).toBe('f32');
        expect(node._shaderCode).toContain('var<uniform> multiplier: f32;');
        expect(node._shaderCode).toContain(
            `var<uniform> ${VOLTEN_INTERNAL_BOUNDS_NAME}: vec4u;`
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

        const node = v.pass(kernel, { data: input });

        expect(node._bounds).toEqual([4, 1, 1]);
        expect(node._dispatch).toEqual([1, 1, 1]);
        expect(node._shaderCode).toContain(VOLTEN_INTERNAL_BOUNDS_NAME);
        expect(node._shaderCode).toContain(
            '_volten_internal_user_main_entrypoint_wrapper'
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

        const node = v.pass(kernel, { data: input });

        expect(
            node._bindingEntries.find(
                (e) => e.name === VOLTEN_INTERNAL_BOUNDS_NAME
            )
        ).toBeUndefined();
        expect(node._shaderCode).not.toContain(VOLTEN_INTERNAL_BOUNDS_NAME);
        expect(node._shaderCode).not.toContain(
            '_volten_internal_user_main_entrypoint_wrapper'
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

        expect(() => v.pass(kernel, { data: input })).toThrow(
            /unsafeManualBounds: true/
        );
    });

    it('handles kernel with no outputs (side-effect only)', () => {
        const v = createMockVoltenContext();
        const data = makeBuffer([1, 2, 3], 'f32', 'rw');

        const kernel = new Kernel('fn main(gid: vec3u) { }', {
            threads: 3
        });

        const node = v.pass(kernel, { data });

        const outputs = getNodeOutputs(node);
        expect(Object.keys(outputs)).toEqual(['data']);
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

        const node = v.pass(kernel, { input, albedo, normal });

        expect(node.albedo._name).toBe('albedo');
        expect(node.normal._name).toBe('normal');

        const outputs = getNodeOutputs(node);
        // All buffer bindings get handles (input too)
        expect(Object.keys(outputs).sort()).toEqual([
            'albedo',
            'input',
            'normal'
        ]);
    });

    it('creates handles for in-place buffer bindings without output declarations', () => {
        const v = createMockVoltenContext();
        const data = makeBuffer([1, 2, 3, 4], 'f32', 'rw');

        // No outputs declared
        const kernel = new Kernel('fn main(gid: vec3u) { }', {
            threads: 4
        });

        const A = v.pass(kernel, { data });

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
        const A = v.pass(kernel, { inout: buf });
        const B = v.pass(kernel, { inout: A.inout });
        const C = v.pass(kernel, { inout: B.inout });

        // C should depend on B, which depends on A
        expect(C._dependencies).toHaveLength(1);
        expect(C._dependencies[0]._id).toBe(B._id);
        expect(B._dependencies).toHaveLength(1);
        expect(B._dependencies[0]._id).toBe(A._id);
    });

    it('Handle type resolves from source buffer (not hardcoded)', () => {
        const v = createMockVoltenContext();
        const positions = new Buffer([[1, 2, 3]], 'vec3f' as any, 'rw');

        const kernel = new Kernel('fn main(gid: vec3u) { }', { threads: 1 });

        const A = v.pass(kernel, { positions });
        const B = v.pass(kernel, { data: A.positions });

        // The binding entry for 'data' should have vec3f type from the source
        const dataEntry = B._bindingEntries.find((e) => e.name === 'data');
        expect(dataEntry).toBeDefined();
        expect(dataEntry!.wgslType).toBe('array<vec3f>');
        expect(dataEntry!.wgslAccess).toBe('read_write');
    });
});

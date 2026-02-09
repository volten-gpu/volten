/**
 * Tests for Kernel class and builtin shorthand expansion.
 */
import { describe, it, expect } from 'vitest';
import {
    Kernel,
    BUILTIN_SHORTHANDS,
    expandParameter,
    expandParameterList,
    expandBuiltinShorthands,
    injectComputeDecorators,
    processShaderSource,
} from '../src/kernel/index.js';

// =============================================================================
// BUILTIN SHORTHAND TESTS
// =============================================================================

describe('Builtin Shorthands', () => {
    describe('BUILTIN_SHORTHANDS constant', () => {
        it('defines gid shorthand', () => {
            expect(BUILTIN_SHORTHANDS.gid).toEqual({
                builtin: 'global_invocation_id',
                type: 'vec3<u32>',
            });
        });

        it('defines lid shorthand', () => {
            expect(BUILTIN_SHORTHANDS.lid).toEqual({
                builtin: 'local_invocation_index',
                type: 'u32',
            });
        });

        it('defines wid shorthand', () => {
            expect(BUILTIN_SHORTHANDS.wid).toEqual({
                builtin: 'workgroup_id',
                type: 'vec3<u32>',
            });
        });

        it('defines lid3 shorthand', () => {
            expect(BUILTIN_SHORTHANDS.lid3).toEqual({
                builtin: 'local_invocation_id',
                type: 'vec3<u32>',
            });
        });

        it('defines nwg shorthand', () => {
            expect(BUILTIN_SHORTHANDS.nwg).toEqual({
                builtin: 'num_workgroups',
                type: 'vec3<u32>',
            });
        });
    });

    describe('expandParameter()', () => {
        it('expands gid shorthand', () => {
            expect(expandParameter('gid: vec3<u32>')).toBe(
                '@builtin(global_invocation_id) gid: vec3<u32>'
            );
        });

        it('expands lid shorthand', () => {
            expect(expandParameter('lid: u32')).toBe(
                '@builtin(local_invocation_index) lid: u32'
            );
        });

        it('expands wid shorthand', () => {
            expect(expandParameter('wid: vec3<u32>')).toBe(
                '@builtin(workgroup_id) wid: vec3<u32>'
            );
        });

        it('expands lid3 shorthand', () => {
            expect(expandParameter('lid3: vec3<u32>')).toBe(
                '@builtin(local_invocation_id) lid3: vec3<u32>'
            );
        });

        it('expands nwg shorthand', () => {
            expect(expandParameter('nwg: vec3<u32>')).toBe(
                '@builtin(num_workgroups) nwg: vec3<u32>'
            );
        });

        it('leaves unknown names untouched', () => {
            expect(expandParameter('index: u32')).toBe('index: u32');
            expect(expandParameter('myParam: vec3<f32>')).toBe('myParam: vec3<f32>');
        });

        it('leaves existing @builtin decorators untouched', () => {
            const decorated = '@builtin(global_invocation_id) gid: vec3<u32>';
            expect(expandParameter(decorated)).toBe(decorated);
        });

        it('handles whitespace variations', () => {
            expect(expandParameter('  gid: vec3<u32>  ')).toBe(
                '@builtin(global_invocation_id) gid: vec3<u32>'
            );
            expect(expandParameter('gid:vec3<u32>')).toBe(
                '@builtin(global_invocation_id) gid: vec3<u32>'
            );
        });

        it('throws on incorrect type for shorthand', () => {
            expect(() => expandParameter('gid: u32')).toThrow(
                /expects type "vec3<u32>"/
            );
            expect(() => expandParameter('lid: vec3<u32>')).toThrow(
                /expects type "u32"/
            );
        });

        it('handles empty string', () => {
            expect(expandParameter('')).toBe('');
            expect(expandParameter('   ')).toBe('');
        });
    });

    describe('expandParameterList()', () => {
        it('expands multiple shorthands', () => {
            expect(expandParameterList('gid: vec3<u32>, lid: u32')).toBe(
                '@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32'
            );
        });

        it('handles mixed shorthand and regular params', () => {
            expect(expandParameterList('gid: vec3<u32>, myBuffer: u32')).toBe(
                '@builtin(global_invocation_id) gid: vec3<u32>, myBuffer: u32'
            );
        });

        it('handles empty parameter list', () => {
            expect(expandParameterList('')).toBe('');
        });
    });

    describe('expandBuiltinShorthands()', () => {
        it('expands shorthands in main function', () => {
            const source = `fn main(gid: vec3<u32>) {
                output[gid.x] = input[gid.x];
            }`;
            const result = expandBuiltinShorthands(source);
            expect(result).toContain('@builtin(global_invocation_id) gid: vec3<u32>');
        });

        it('only processes main function', () => {
            const source = `
fn helper(gid: vec3<u32>) { }
fn main(gid: vec3<u32>) { }
`;
            const result = expandBuiltinShorthands(source);
            // main should be expanded
            expect(result).toContain('fn main(@builtin(global_invocation_id) gid: vec3<u32>)');
            // helper should NOT be expanded
            expect(result).toContain('fn helper(gid: vec3<u32>)');
        });

        it('handles complex parameter lists', () => {
            const source = `fn main(gid: vec3<u32>, lid: u32, wid: vec3<u32>) {
                // compute logic
            }`;
            const result = expandBuiltinShorthands(source);
            expect(result).toContain('@builtin(global_invocation_id) gid: vec3<u32>');
            expect(result).toContain('@builtin(local_invocation_index) lid: u32');
            expect(result).toContain('@builtin(workgroup_id) wid: vec3<u32>');
        });
    });
});

// =============================================================================
// DECORATOR INJECTION TESTS
// =============================================================================

describe('Compute Decorator Injection', () => {
    describe('injectComputeDecorators()', () => {
        it('injects @compute and @workgroup_size', () => {
            const source = `fn main(gid: vec3<u32>) { }`;
            const result = injectComputeDecorators(source, [64, 1, 1]);
            expect(result).toBe(`@compute @workgroup_size(64, 1, 1)
fn main(gid: vec3<u32>) { }`);
        });

        it('uses default workgroup size', () => {
            const source = `fn main(gid: vec3<u32>) { }`;
            const result = injectComputeDecorators(source);
            expect(result).toContain('@workgroup_size(64, 1, 1)');
        });

        it('supports custom workgroup size', () => {
            const source = `fn main() { }`;
            const result = injectComputeDecorators(source, [256, 4, 2]);
            expect(result).toContain('@workgroup_size(256, 4, 2)');
        });
    });

    describe('processShaderSource()', () => {
        it('combines expansion and injection', () => {
            const source = `fn main(gid: vec3<u32>) {
    output[gid.x] = input[gid.x] * 2.0;
}`;
            const result = processShaderSource(source, [128, 1, 1]);
            expect(result).toContain('@compute @workgroup_size(128, 1, 1)');
            expect(result).toContain('@builtin(global_invocation_id) gid: vec3<u32>');
        });
    });
});

// =============================================================================
// KERNEL CLASS TESTS
// =============================================================================

describe('Kernel Class', () => {
    describe('constructor', () => {
        it('stores source code', () => {
            const kernel = new Kernel('fn main() { }');
            expect(kernel.source).toBe('fn main() { }');
        });

        it('defaults to empty outputs', () => {
            const kernel = new Kernel('fn main() { }');
            expect(kernel.outputs).toEqual([]);
        });

        it('defaults to [64, 1, 1] workgroup size', () => {
            const kernel = new Kernel('fn main() { }');
            expect(kernel.workgroupSize).toEqual([64, 1, 1]);
        });
    });

    describe('outputs normalization', () => {
        it('handles string array outputs', () => {
            const kernel = new Kernel('fn main() { }', {
                outputs: ['result', 'debug'],
            });
            expect(kernel.outputs).toEqual([
                { name: 'result' },
                { name: 'debug' },
            ]);
            expect(kernel.outputNames).toEqual(['result', 'debug']);
        });

        it('handles object outputs with size', () => {
            const kernel = new Kernel('fn main() { }', {
                outputs: {
                    result: { size: 1 },
                    partial: { size: (data) => (data.input as any).length / 2 },
                },
            });
            expect(kernel.outputs).toHaveLength(2);
            expect(kernel.outputs[0].name).toBe('result');
            expect(kernel.outputs[0].size).toBe(1);
            expect(kernel.outputs[1].name).toBe('partial');
            expect(typeof kernel.outputs[1].size).toBe('function');
        });
    });

    describe('workgroupSize normalization', () => {
        it('normalizes partial tuple', () => {
            const kernel = new Kernel('fn main() { }', {
                workgroupSize: [256],
            });
            expect(kernel.workgroupSize).toEqual([256, 1, 1]);
        });

        it('normalizes two-element tuple', () => {
            const kernel = new Kernel('fn main() { }', {
                workgroupSize: [16, 16],
            });
            expect(kernel.workgroupSize).toEqual([16, 16, 1]);
        });

        it('passes through full tuple', () => {
            const kernel = new Kernel('fn main() { }', {
                workgroupSize: [4, 4, 4],
            });
            expect(kernel.workgroupSize).toEqual([4, 4, 4]);
        });
    });

    describe('threads option', () => {
        it('stores number threads spec', () => {
            const kernel = new Kernel('fn main() { }', { threads: 1024 });
            expect(kernel.threads).toBe(1024);
        });

        it('stores string threads spec', () => {
            const kernel = new Kernel('fn main() { }', { threads: 'input' });
            expect(kernel.threads).toBe('input');
        });

        it('stores function threads spec', () => {
            const fn = (data: Record<string, unknown>) => [100, 1, 1] as [number, number, number];
            const kernel = new Kernel('fn main() { }', { threads: fn });
            expect(kernel.threads).toBe(fn);
        });
    });

    describe('assembledSource', () => {
        it('expands shorthands and injects decorators', () => {
            const kernel = new Kernel(`
fn main(gid: vec3<u32>) {
    output[gid.x] = input[gid.x];
}
`, { workgroupSize: [128] });

            expect(kernel.assembledSource).toContain('@compute @workgroup_size(128, 1, 1)');
            expect(kernel.assembledSource).toContain('@builtin(global_invocation_id) gid: vec3<u32>');
        });

        it('caches assembled source', () => {
            const kernel = new Kernel('fn main() { }');
            const first = kernel.assembledSource;
            const second = kernel.assembledSource;
            expect(first).toBe(second); // Same object reference
        });

        it('preserves original source', () => {
            const original = 'fn main(gid: vec3<u32>) { }';
            const kernel = new Kernel(original);
            expect(kernel.source).toBe(original);
            expect(kernel.assembledSource).not.toBe(original);
        });
    });
});

/**
 * Comprehensive tests for buffer packing logic.
 * 
 * These tests verify that JavaScript data is correctly packed into
 * WGSL-compatible binary format with proper alignment and padding.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { pack, getStride, roundUp, getByteLength } from '../src/utils/alignment.js';
import { struct, array, resolveType, getWgslType } from '../src/types/schema.js';
import { PRIMITIVE_INFO, isPrimitiveType, getPrimitiveInfo } from '../src/types/primitives.js';

/**
 * Helper to read float values from packed buffer
 */
function readFloats(buffer: ArrayBuffer, count: number): number[] {
    const view = new Float32Array(buffer);
    return Array.from(view.slice(0, count));
}

/**
 * Helper to read uint32 values from packed buffer
 */
function readUints(buffer: ArrayBuffer, count: number): number[] {
    const view = new Uint32Array(buffer);
    return Array.from(view.slice(0, count));
}

/**
 * Helper to read int32 values from packed buffer
 */
function readInts(buffer: ArrayBuffer, count: number): number[] {
    const view = new Int32Array(buffer);
    return Array.from(view.slice(0, count));
}

/**
 * Helper to read raw bytes from packed buffer
 */
function readBytes(buffer: ArrayBuffer): number[] {
    return Array.from(new Uint8Array(buffer));
}

// =============================================================================
// SCALAR TYPE TESTS
// =============================================================================

describe('Scalar Packing', () => {
    describe('f32 (float)', () => {
        it('packs a single f32 value', () => {
            const packed = pack([3.14], 'f32');
            expect(packed.byteLength).toBe(4);
            expect(readFloats(packed, 1)[0]).toBeCloseTo(3.14, 5);
        });

        it('packs multiple f32 values', () => {
            const packed = pack([1.0, 2.0, 3.0], 'f32');
            expect(packed.byteLength).toBe(12); // 3 * 4 bytes
            expect(readFloats(packed, 3)).toEqual([1.0, 2.0, 3.0]);
        });

        it('handles negative and zero values', () => {
            const packed = pack([-5.5, 0, 100.25], 'f32');
            const values = readFloats(packed, 3);
            expect(values[0]).toBeCloseTo(-5.5, 5);
            expect(values[1]).toBe(0);
            expect(values[2]).toBeCloseTo(100.25, 5);
        });
    });

    describe('u32 (unsigned int)', () => {
        it('packs a single u32 value', () => {
            const packed = pack([42], 'u32');
            expect(packed.byteLength).toBe(4);
            expect(readUints(packed, 1)).toEqual([42]);
        });

        it('packs multiple u32 values', () => {
            const packed = pack([0, 255, 65535, 4294967295], 'u32');
            expect(packed.byteLength).toBe(16);
            expect(readUints(packed, 4)).toEqual([0, 255, 65535, 4294967295]);
        });
    });

    describe('i32 (signed int)', () => {
        it('packs positive and negative values', () => {
            const packed = pack([-100, 0, 100], 'i32');
            expect(packed.byteLength).toBe(12);
            expect(readInts(packed, 3)).toEqual([-100, 0, 100]);
        });

        it('handles min/max i32 values', () => {
            const packed = pack([-2147483648, 2147483647], 'i32');
            expect(readInts(packed, 2)).toEqual([-2147483648, 2147483647]);
        });
    });

    describe('bool', () => {
        it('packs boolean as u32 (WGSL storage bool is 4 bytes)', () => {
            const packed = pack([true, false, true], 'bool');
            expect(packed.byteLength).toBe(12);
            expect(readUints(packed, 3)).toEqual([1, 0, 1]);
        });
    });
});

// =============================================================================
// VECTOR TYPE TESTS
// =============================================================================

describe('Vector Packing', () => {
    describe('vec2f', () => {
        it('packs a single vec2f', () => {
            const packed = pack([[1.0, 2.0]], 'vec2f');
            expect(packed.byteLength).toBe(8); // 2 * 4 bytes
            expect(readFloats(packed, 2)).toEqual([1.0, 2.0]);
        });

        it('packs multiple vec2f values', () => {
            const packed = pack([[1.0, 2.0], [3.0, 4.0]], 'vec2f');
            expect(packed.byteLength).toBe(16);
            expect(readFloats(packed, 4)).toEqual([1.0, 2.0, 3.0, 4.0]);
        });

        it('has stride of 8 bytes', () => {
            expect(getStride('vec2f')).toBe(8);
        });
    });

    describe('vec3f (critical alignment test)', () => {
        it('packs a single vec3f with correct size', () => {
            const packed = pack([[1.0, 2.0, 3.0]], 'vec3f');
            // vec3f is 12 bytes data, but stride is 16 due to alignment
            expect(packed.byteLength).toBe(16);
        });

        it('has stride of 16 bytes (not 12!) due to WGSL alignment rules', () => {
            expect(getStride('vec3f')).toBe(16);
            expect(PRIMITIVE_INFO.vec3f.size).toBe(12);
            expect(PRIMITIVE_INFO.vec3f.alignment).toBe(16);
        });

        it('correctly packs vec3f data in first 12 bytes', () => {
            const packed = pack([[1.0, 2.0, 3.0]], 'vec3f');
            const values = readFloats(packed, 3);
            expect(values).toEqual([1.0, 2.0, 3.0]);
        });

        it('packs multiple vec3f with 16-byte stride', () => {
            const packed = pack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 'vec3f');
            expect(packed.byteLength).toBe(32); // 2 * 16 bytes stride

            const view = new Float32Array(packed);
            // First vec3f at offset 0-12
            expect(view[0]).toBe(1.0);
            expect(view[1]).toBe(2.0);
            expect(view[2]).toBe(3.0);
            // view[3] is padding
            // Second vec3f at offset 16-28
            expect(view[4]).toBe(4.0);
            expect(view[5]).toBe(5.0);
            expect(view[6]).toBe(6.0);
        });
    });

    describe('vec4f', () => {
        it('packs vec4f values correctly', () => {
            const packed = pack([[1.0, 2.0, 3.0, 4.0]], 'vec4f');
            expect(packed.byteLength).toBe(16);
            expect(readFloats(packed, 4)).toEqual([1.0, 2.0, 3.0, 4.0]);
        });

        it('has stride of 16 bytes (no padding needed)', () => {
            expect(getStride('vec4f')).toBe(16);
        });
    });

    describe('integer vectors', () => {
        it('packs vec2u correctly', () => {
            const packed = pack([[10, 20]], 'vec2u');
            expect(readUints(packed, 2)).toEqual([10, 20]);
        });

        it('packs vec3i correctly with 16-byte stride', () => {
            const packed = pack([[-1, 0, 1]], 'vec3i');
            expect(packed.byteLength).toBe(16);
            expect(readInts(packed, 3)).toEqual([-1, 0, 1]);
        });

        it('packs vec4u correctly', () => {
            const packed = pack([[1, 2, 3, 4]], 'vec4u');
            expect(readUints(packed, 4)).toEqual([1, 2, 3, 4]);
        });
    });
});

// =============================================================================
// MATRIX TYPE TESTS
// =============================================================================

describe('Matrix Packing', () => {
    describe('mat4x4f (most common case)', () => {
        it('has correct size and alignment', () => {
            expect(PRIMITIVE_INFO.mat4x4f.size).toBe(64);
            expect(PRIMITIVE_INFO.mat4x4f.alignment).toBe(16);
        });

        it('packs identity matrix correctly (column-major)', () => {
            // Identity matrix as flat column-major array
            const identity = [
                1, 0, 0, 0,  // Column 0
                0, 1, 0, 0,  // Column 1
                0, 0, 1, 0,  // Column 2
                0, 0, 0, 1,  // Column 3
            ];
            const packed = pack([identity], 'mat4x4f');
            expect(packed.byteLength).toBe(64);
            expect(readFloats(packed, 16)).toEqual(identity);
        });

        it('packs matrix with meaningful values', () => {
            const matrix = [
                1, 2, 3, 4,     // Column 0
                5, 6, 7, 8,     // Column 1
                9, 10, 11, 12,  // Column 2
                13, 14, 15, 16, // Column 3
            ];
            const packed = pack([matrix], 'mat4x4f');
            expect(readFloats(packed, 16)).toEqual(matrix);
        });
    });

    describe('mat3x3f', () => {
        it('has size 48 (3 columns * 16 bytes each due to vec3f alignment)', () => {
            expect(PRIMITIVE_INFO.mat3x3f.size).toBe(48);
            expect(PRIMITIVE_INFO.mat3x3f.alignment).toBe(16);
        });

        it('packs mat3x3f with column padding', () => {
            // 9 values, but each column of 3 values needs 16 bytes
            const matrix = [
                1, 2, 3,     // Column 0 (needs 16 bytes)
                4, 5, 6,     // Column 1 (needs 16 bytes)
                7, 8, 9,     // Column 2 (needs 16 bytes)
            ];
            const packed = pack([matrix], 'mat3x3f');
            expect(packed.byteLength).toBe(48);

            const view = new Float32Array(packed);
            // Column 0 at bytes 0-12 (indices 0-2), padding at index 3
            expect(view[0]).toBe(1);
            expect(view[1]).toBe(2);
            expect(view[2]).toBe(3);
            // Column 1 at bytes 16-28 (indices 4-6)
            expect(view[4]).toBe(4);
            expect(view[5]).toBe(5);
            expect(view[6]).toBe(6);
            // Column 2 at bytes 32-44 (indices 8-10)
            expect(view[8]).toBe(7);
            expect(view[9]).toBe(8);
            expect(view[10]).toBe(9);
        });
    });

    describe('mat2x2f', () => {
        it('has size 16 (2 columns * 8 bytes each)', () => {
            expect(PRIMITIVE_INFO.mat2x2f.size).toBe(16);
            expect(PRIMITIVE_INFO.mat2x2f.alignment).toBe(8);
        });
    });
});

// =============================================================================
// STRUCT TESTS
// =============================================================================

describe('Struct Schema', () => {
    describe('struct() creation', () => {
        it('creates a simple struct with correct layout', () => {
            const Simple = struct({ x: 'f32', y: 'f32' });
            expect(Simple.kind).toBe('struct');
            expect(Simple.fields).toHaveLength(2);
            expect(Simple.size).toBe(8);
            expect(Simple.alignment).toBe(4);
        });

        it('calculates field offsets correctly', () => {
            const Point = struct({ x: 'f32', y: 'f32', z: 'f32' });
            expect(Point.fields[0].offset).toBe(0);
            expect(Point.fields[1].offset).toBe(4);
            expect(Point.fields[2].offset).toBe(8);
        });

        it('handles alignment padding between fields', () => {
            // f32 followed by vec3f: the vec3f needs 16-byte alignment
            const Mixed = struct({
                scalar: 'f32',   // 4 bytes at offset 0
                vector: 'vec3f'  // needs 16-byte alignment, so offset 16
            });
            expect(Mixed.fields[0].offset).toBe(0);
            expect(Mixed.fields[1].offset).toBe(16); // Padded to 16-byte boundary
            expect(Mixed.size).toBe(32); // 16 + 12 = 28, rounded up to 32 for alignment
        });

        it('stores optional name', () => {
            const Named = struct({ x: 'f32' }, 'MyStruct');
            expect(Named.name).toBe('MyStruct');
        });
    });

    describe('Particle struct (realistic use case)', () => {
        const Particle = struct({
            position: 'vec3f',  // 12 bytes, 16-byte aligned -> offset 0
            velocity: 'vec3f',  // 12 bytes, 16-byte aligned -> offset 16
            mass: 'f32',        // 4 bytes, 4-byte aligned -> offset 28
        });

        it('has correct total size with alignment', () => {
            // position: 0-12, velocity: 16-28, mass: 28-32
            // Total: 32 bytes, struct alignment: 16
            expect(Particle.size).toBe(32);
            expect(Particle.alignment).toBe(16);
        });

        it('packs particle data correctly', () => {
            const data = [
                { position: [1, 2, 3], velocity: [4, 5, 6], mass: 10.0 }
            ];
            const packed = pack(data, Particle);
            expect(packed.byteLength).toBe(32);

            const view = new Float32Array(packed);
            // position at offset 0
            expect(view[0]).toBe(1);
            expect(view[1]).toBe(2);
            expect(view[2]).toBe(3);
            // velocity at offset 16 (index 4)
            expect(view[4]).toBe(4);
            expect(view[5]).toBe(5);
            expect(view[6]).toBe(6);
            // mass at offset 28 (index 7)
            expect(view[7]).toBe(10.0);
        });

        it('packs multiple particles with correct stride', () => {
            const data = [
                { position: [1, 0, 0], velocity: [0, 1, 0], mass: 1.0 },
                { position: [2, 0, 0], velocity: [0, 2, 0], mass: 2.0 }
            ];
            const packed = pack(data, Particle);
            expect(packed.byteLength).toBe(64); // 2 * 32 bytes

            const view = new Float32Array(packed);
            // Second particle starts at index 8 (32 bytes / 4)
            expect(view[8]).toBe(2); // position.x
            expect(view[15]).toBe(2.0); // mass at index 8+7
        });
    });

    describe('Nested structs', () => {
        const AABB = struct({
            min: 'vec3f',
            max: 'vec3f',
        });

        const Entity = struct({
            position: 'vec3f',
            bounds: AABB,
            id: 'u32',
        });

        it('calculates nested struct layout correctly', () => {
            // AABB: min at 0-12, max at 16-28, size 32, align 16
            expect(AABB.size).toBe(32);
            expect(AABB.alignment).toBe(16);

            // Entity: position at 0-12 (align 16)
            //         bounds at 16 (AABB needs 16-align), size 32
            //         id at 48, size 4
            //         Total: 52, rounded to 64 for alignment
            expect(Entity.fields[0].offset).toBe(0);  // position
            expect(Entity.fields[1].offset).toBe(16); // bounds (AABB)
            expect(Entity.fields[2].offset).toBe(48); // id
        });

        it('packs nested struct data correctly', () => {
            const data = [{
                position: [10, 20, 30],
                bounds: { min: [0, 0, 0], max: [1, 1, 1] },
                id: 42,
            }];
            const packed = pack(data, Entity);

            const floatView = new Float32Array(packed);
            const uintView = new Uint32Array(packed);

            // position at offset 0
            expect(floatView[0]).toBe(10);
            expect(floatView[1]).toBe(20);
            expect(floatView[2]).toBe(30);

            // bounds.min at offset 16 (index 4)
            expect(floatView[4]).toBe(0);
            expect(floatView[5]).toBe(0);
            expect(floatView[6]).toBe(0);

            // bounds.max at offset 32 (index 8)
            expect(floatView[8]).toBe(1);
            expect(floatView[9]).toBe(1);
            expect(floatView[10]).toBe(1);

            // id at offset 48 (index 12)
            expect(uintView[12]).toBe(42);
        });
    });
});

// =============================================================================
// FIXED-SIZE ARRAY TESTS
// =============================================================================

describe('Array Schema', () => {
    describe('array() creation', () => {
        it('creates array of primitives', () => {
            const arr = array('f32', 10);
            expect(arr.kind).toBe('array');
            expect(arr.count).toBe(10);
            expect(arr.stride).toBe(4);
            expect(arr.size).toBe(40);
        });

        it('creates array of vectors with correct stride', () => {
            const arr = array('vec3f', 5);
            expect(arr.stride).toBe(16); // vec3f has 16-byte stride
            expect(arr.size).toBe(80);
            expect(arr.alignment).toBe(16);
        });

        it('throws for invalid count', () => {
            expect(() => array('f32', 0)).toThrow();
            expect(() => array('f32', -1)).toThrow();
            expect(() => array('f32', 1.5)).toThrow();
        });
    });

    describe('struct with fixed-size array field', () => {
        const GridCell = struct({
            neighbors: array('u32', 8),
            temperature: 'f32',
        });

        it('calculates layout with array field', () => {
            // neighbors: 8 * 4 = 32 bytes at offset 0
            // temperature: 4 bytes at offset 32
            // Total: 36, alignment 4
            expect(GridCell.fields[0].offset).toBe(0);
            expect(GridCell.fields[0].size).toBe(32);
            expect(GridCell.fields[1].offset).toBe(32);
            expect(GridCell.size).toBe(36);
        });

        it('packs struct with array field correctly', () => {
            const data = [{
                neighbors: [0, 1, 2, 3, 4, 5, 6, 7],
                temperature: 98.6,
            }];
            const packed = pack(data, GridCell);

            const uintView = new Uint32Array(packed);
            const floatView = new Float32Array(packed);

            expect(uintView[0]).toBe(0);
            expect(uintView[7]).toBe(7);
            expect(floatView[8]).toBeCloseTo(98.6, 1);
        });
    });

    describe('nested arrays', () => {
        const matrix3x3 = array(array('f32', 3), 3);

        it('creates nested array schema', () => {
            expect(matrix3x3.count).toBe(3);
            expect(matrix3x3.stride).toBe(12); // inner array of 3 f32s
            expect(matrix3x3.size).toBe(36);
        });
    });
});

// =============================================================================
// UTILITY FUNCTION TESTS
// =============================================================================

describe('Utility Functions', () => {
    describe('roundUp()', () => {
        it('rounds up to alignment boundary', () => {
            expect(roundUp(0, 16)).toBe(0);
            expect(roundUp(1, 16)).toBe(16);
            expect(roundUp(15, 16)).toBe(16);
            expect(roundUp(16, 16)).toBe(16);
            expect(roundUp(17, 16)).toBe(32);
        });

        it('handles various alignments', () => {
            expect(roundUp(5, 4)).toBe(8);
            expect(roundUp(12, 8)).toBe(16);
            expect(roundUp(7, 1)).toBe(7);
        });
    });

    describe('getStride()', () => {
        it('returns correct stride for primitives', () => {
            expect(getStride('f32')).toBe(4);
            expect(getStride('vec2f')).toBe(8);
            expect(getStride('vec3f')).toBe(16); // Important: not 12!
            expect(getStride('vec4f')).toBe(16);
            expect(getStride('mat4x4f')).toBe(64);
        });

        it('returns correct stride for structs', () => {
            const Simple = struct({ a: 'f32', b: 'f32' });
            expect(getStride(Simple)).toBe(8);

            const WithVec3 = struct({ v: 'vec3f' });
            expect(getStride(WithVec3)).toBe(16);
        });
    });

    describe('getByteLength()', () => {
        it('calculates total byte length correctly', () => {
            expect(getByteLength(10, 'f32')).toBe(40);
            expect(getByteLength(5, 'vec3f')).toBe(80); // 5 * 16
            expect(getByteLength(2, 'mat4x4f')).toBe(128);
        });
    });

    describe('resolveType()', () => {
        it('resolves primitive types', () => {
            const resolved = resolveType('f32');
            expect(resolved.size).toBe(4);
            expect(resolved.alignment).toBe(4);
        });

        it('resolves struct types', () => {
            const S = struct({ x: 'vec3f' });
            const resolved = resolveType(S);
            expect(resolved.size).toBe(S.size);
            expect(resolved.alignment).toBe(S.alignment);
        });

        it('throws for unknown type', () => {
            expect(() => resolveType('unknown_type' as any)).toThrow();
        });
    });

    describe('getWgslType()', () => {
        it('returns primitive type names directly', () => {
            expect(getWgslType('f32')).toBe('f32');
            expect(getWgslType('vec3f')).toBe('vec3f');
            expect(getWgslType('mat4x4f')).toBe('mat4x4f');
        });

        it('returns named struct', () => {
            const Named = struct({ x: 'f32' }, 'MyStruct');
            expect(getWgslType(Named)).toBe('MyStruct');
        });

        it('throws for anonymous struct', () => {
            const Anonymous = struct({ x: 'f32' });
            expect(() => getWgslType(Anonymous)).toThrow();
        });

        it('returns array type string', () => {
            const arr = array('vec3f', 10);
            expect(getWgslType(arr)).toBe('array<vec3f, 10>');
        });
    });

    describe('isPrimitiveType()', () => {
        it('returns true for valid primitives', () => {
            expect(isPrimitiveType('f32')).toBe(true);
            expect(isPrimitiveType('vec3f')).toBe(true);
            expect(isPrimitiveType('mat4x4f')).toBe(true);
        });

        it('returns false for invalid strings', () => {
            expect(isPrimitiveType('float')).toBe(false);
            expect(isPrimitiveType('Vector3')).toBe(false);
            expect(isPrimitiveType('')).toBe(false);
        });
    });

    describe('getPrimitiveInfo()', () => {
        it('returns correct info for all vector types', () => {
            expect(getPrimitiveInfo('vec2f').components).toBe(2);
            expect(getPrimitiveInfo('vec3f').components).toBe(3);
            expect(getPrimitiveInfo('vec4f').components).toBe(4);
        });
    });
});

// =============================================================================
// EDGE CASES AND ERROR HANDLING
// =============================================================================

describe('Edge Cases', () => {
    it('handles empty array', () => {
        const packed = pack([], 'f32');
        expect(packed.byteLength).toBe(0);
    });

    it('handles missing struct fields (fills with zeros)', () => {
        const Point = struct({ x: 'f32', y: 'f32', z: 'f32' });
        // Only provide x, missing y and z
        const packed = pack([{ x: 5.0 }], Point);
        const values = readFloats(packed, 3);
        expect(values[0]).toBe(5.0);
        expect(values[1]).toBe(0); // default
        expect(values[2]).toBe(0); // default
    });

    it('handles partial vector (fills missing components with 0)', () => {
        // Provide only 2 components to vec4f
        const packed = pack([[1.0, 2.0]], 'vec4f');
        const values = readFloats(packed, 4);
        expect(values).toEqual([1.0, 2.0, 0, 0]);
    });

    it('handles typed arrays as input', () => {
        const input = new Float32Array([1.0, 2.0, 3.0]);
        const packed = pack(Array.from(input), 'f32');
        expect(readFloats(packed, 3)).toEqual([1.0, 2.0, 3.0]);
    });
});

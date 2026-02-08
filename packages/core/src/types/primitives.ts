/**
 * Primitive WGSL type names
 */
export type ScalarType = 'f32' | 'u32' | 'i32' | 'f16' | 'bool';
export type VectorType =
    | 'vec2f' | 'vec3f' | 'vec4f'
    | 'vec2u' | 'vec3u' | 'vec4u'
    | 'vec2i' | 'vec3i' | 'vec4i'
    | 'vec2h' | 'vec3h' | 'vec4h';
export type MatrixType =
    | 'mat2x2f' | 'mat2x3f' | 'mat2x4f'
    | 'mat3x2f' | 'mat3x3f' | 'mat3x4f'
    | 'mat4x2f' | 'mat4x3f' | 'mat4x4f';

export type PrimitiveType = ScalarType | VectorType | MatrixType;

/**
 * Type metadata: size in bytes and alignment in bytes
 */
export interface TypeInfo {
    size: number;
    alignment: number;
    /** Number of components (1 for scalars, 2-4 for vectors, cols*rows for matrices) */
    components: number;
    /** Base type for typed array selection */
    baseType: 'f32' | 'u32' | 'i32' | 'f16';
}

/**
 * WGSL type information lookup table
 * Based on WGSL spec alignment rules:
 * - Scalars: 4 bytes, 4-byte aligned (f16 is 2 bytes, 2-byte aligned)
 * - vec2: 8 bytes, 8-byte aligned
 * - vec3: 12 bytes, 16-byte aligned (!)
 * - vec4: 16 bytes, 16-byte aligned
 * - matNxM: N columns of vecM, each column 16-byte aligned
 */
export const PRIMITIVE_INFO: Record<PrimitiveType, TypeInfo> = {
    // Scalars
    f32: { size: 4, alignment: 4, components: 1, baseType: 'f32' },
    u32: { size: 4, alignment: 4, components: 1, baseType: 'u32' },
    i32: { size: 4, alignment: 4, components: 1, baseType: 'i32' },
    f16: { size: 2, alignment: 2, components: 1, baseType: 'f16' },
    bool: { size: 4, alignment: 4, components: 1, baseType: 'u32' }, // WGSL bool is 4 bytes in storage

    // Float vectors
    vec2f: { size: 8, alignment: 8, components: 2, baseType: 'f32' },
    vec3f: { size: 12, alignment: 16, components: 3, baseType: 'f32' },
    vec4f: { size: 16, alignment: 16, components: 4, baseType: 'f32' },

    // Unsigned int vectors
    vec2u: { size: 8, alignment: 8, components: 2, baseType: 'u32' },
    vec3u: { size: 12, alignment: 16, components: 3, baseType: 'u32' },
    vec4u: { size: 16, alignment: 16, components: 4, baseType: 'u32' },

    // Signed int vectors
    vec2i: { size: 8, alignment: 8, components: 2, baseType: 'i32' },
    vec3i: { size: 12, alignment: 16, components: 3, baseType: 'i32' },
    vec4i: { size: 16, alignment: 16, components: 4, baseType: 'i32' },

    // Half-float vectors
    vec2h: { size: 4, alignment: 4, components: 2, baseType: 'f16' },
    vec3h: { size: 6, alignment: 8, components: 3, baseType: 'f16' },
    vec4h: { size: 8, alignment: 8, components: 4, baseType: 'f16' },

    // Matrices (column-major, each column is a vec)
    // mat2x2f = 2 columns of vec2f
    mat2x2f: { size: 16, alignment: 8, components: 4, baseType: 'f32' },
    // mat2x3f = 2 columns of vec3f (each 16-byte aligned)
    mat2x3f: { size: 32, alignment: 16, components: 6, baseType: 'f32' },
    // mat2x4f = 2 columns of vec4f
    mat2x4f: { size: 32, alignment: 16, components: 8, baseType: 'f32' },
    // mat3x2f = 3 columns of vec2f
    mat3x2f: { size: 24, alignment: 8, components: 6, baseType: 'f32' },
    // mat3x3f = 3 columns of vec3f (each 16-byte aligned, so 48 bytes total)
    mat3x3f: { size: 48, alignment: 16, components: 9, baseType: 'f32' },
    // mat3x4f = 3 columns of vec4f
    mat3x4f: { size: 48, alignment: 16, components: 12, baseType: 'f32' },
    // mat4x2f = 4 columns of vec2f
    mat4x2f: { size: 32, alignment: 8, components: 8, baseType: 'f32' },
    // mat4x3f = 4 columns of vec3f (each 16-byte aligned, so 64 bytes total)
    mat4x3f: { size: 64, alignment: 16, components: 12, baseType: 'f32' },
    // mat4x4f = 4 columns of vec4f
    mat4x4f: { size: 64, alignment: 16, components: 16, baseType: 'f32' },
};

/**
 * Check if a string is a known primitive type
 */
export function isPrimitiveType(type: string): type is PrimitiveType {
    return type in PRIMITIVE_INFO;
}

/**
 * Get type info for a primitive type
 */
export function getPrimitiveInfo(type: PrimitiveType): TypeInfo {
    return PRIMITIVE_INFO[type];
}

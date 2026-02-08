// struct() and array() implementations
// Type schema builders for complex WGSL types

import { type PrimitiveType, isPrimitiveType, getPrimitiveInfo, type TypeInfo } from './primitives.js';

/**
 * Schema types for type resolution
 */
export type TypeDescriptor = PrimitiveType | StructSchema | ArraySchema;

/**
 * Field definition within a struct
 */
export interface FieldInfo {
    name: string;
    type: TypeDescriptor;
    offset: number;
    size: number;
    alignment: number;
}

/**
 * Struct schema with computed layout
 */
export interface StructSchema {
    readonly kind: 'struct';
    readonly fields: FieldInfo[];
    readonly size: number;
    readonly alignment: number;
    /** Name for WGSL type generation */
    readonly name: string;
}

/**
 * Fixed-size array schema
 */
export interface ArraySchema {
    readonly kind: 'array';
    readonly elementType: TypeDescriptor;
    readonly count: number;
    readonly stride: number;
    readonly size: number;
    readonly alignment: number;
}

/**
 * Resolved type info (what we need for packing)
 */
export interface ResolvedType {
    size: number;
    alignment: number;
    descriptor: TypeDescriptor;
}

/**
 * Round up to alignment boundary
 */
function roundUp(value: number, alignment: number): number {
    return Math.ceil(value / alignment) * alignment;
}

/**
 * Resolve any type descriptor to size and alignment
 */
export function resolveType(type: TypeDescriptor): ResolvedType {
    if (typeof type === 'string') {
        if (!isPrimitiveType(type)) {
            throw new Error(`Unknown primitive type: "${type}"`);
        }
        const info = getPrimitiveInfo(type);
        return { size: info.size, alignment: info.alignment, descriptor: type };
    }

    if (type.kind === 'struct') {
        return { size: type.size, alignment: type.alignment, descriptor: type };
    }

    if (type.kind === 'array') {
        return { size: type.size, alignment: type.alignment, descriptor: type };
    }

    throw new Error('Invalid type descriptor');
}

/**
 * Create a struct schema from field definitions
 * 
 * @example
 * const Particle = struct("Particle", {
 *   position: "vec3f",
 *   velocity: "vec3f",
 *   mass: "f32",
 * });
 */
export function struct(
    name: string,
    fields: Record<string, TypeDescriptor>
): StructSchema {
    const fieldInfos: FieldInfo[] = [];
    let offset = 0;
    let maxAlignment = 0;

    for (const [fieldName, fieldType] of Object.entries(fields)) {
        const resolved = resolveType(fieldType);

        // Align offset to field's alignment requirement
        offset = roundUp(offset, resolved.alignment);

        fieldInfos.push({
            name: fieldName,
            type: fieldType,
            offset,
            size: resolved.size,
            alignment: resolved.alignment,
        });

        offset += resolved.size;
        maxAlignment = Math.max(maxAlignment, resolved.alignment);
    }

    // Struct size must be rounded up to alignment for array of struct
    const size = roundUp(offset, maxAlignment);

    return {
        kind: 'struct',
        fields: fieldInfos,
        size,
        alignment: maxAlignment,
        name,
    };
}

/**
 * Create a fixed-size array schema
 * 
 * @example
 * const positions = array("vec3f", 100);
 * const nested = array(array("f32", 4), 8);
 */
export function array(elementType: TypeDescriptor, count: number): ArraySchema {
    if (count <= 0 || !Number.isInteger(count)) {
        throw new Error(`Array count must be a positive integer, got: ${count}`);
    }

    const resolved = resolveType(elementType);

    // Array stride is element size rounded up to element alignment
    const stride = roundUp(resolved.size, resolved.alignment);
    const size = stride * count;

    return {
        kind: 'array',
        elementType,
        count,
        stride,
        size,
        alignment: resolved.alignment,
    };
}

/**
 * Get the WGSL type string for a descriptor
 */
export function getWgslType(type: TypeDescriptor): string {
    if (typeof type === 'string') {
        return type;
    }

    if (type.kind === 'struct') {
        return type.name;
    }

    if (type.kind === 'array') {
        const elementWgsl = getWgslType(type.elementType);
        return `array<${elementWgsl}, ${type.count}>`;
    }

    throw new Error('Invalid type descriptor');
}

// struct() and array() implementations
// Type schema builders for complex WGSL types

import { type PrimitiveType, isPrimitiveType } from './primitives.js';

/**
 * Declarative type description used throughout Volten.
 *
 * Concrete offsets, stride, and final byte size are derived later by the
 * layout engine in `utils/layout.ts`.
 */
export type TypeDescriptor = PrimitiveType | StructSchema | ArraySchema;

/**
 * Declarative field definition within a struct schema.
 */
export interface StructField {
    readonly name: string;
    readonly type: TypeDescriptor;
}

/**
 * Declarative struct schema.
 */
export interface StructSchema {
    readonly kind: 'struct';
    readonly fields: readonly StructField[];
    /** Name for WGSL type generation */
    readonly name: string;
}

/**
 * Declarative fixed-size array schema.
 */
export interface ArraySchema {
    readonly kind: 'array';
    readonly elementType: TypeDescriptor;
    readonly count: number;
}

/**
 * Validate a descriptor eagerly so invalid nested schemas fail at construction
 * time instead of later during packing or shader generation.
 */
function assertValidTypeDescriptor(type: TypeDescriptor): void {
    if (typeof type === 'string') {
        if (!isPrimitiveType(type)) {
            throw new Error(`Unknown primitive type: "${type}"`);
        }
        return;
    }

    if (type.kind === 'array') {
        if (type.count <= 0 || !Number.isInteger(type.count)) {
            throw new Error(
                `Array count must be a positive integer, got: ${type.count}`
            );
        }
        assertValidTypeDescriptor(type.elementType);
        return;
    }

    if (type.kind === 'struct') {
        for (const field of type.fields) {
            assertValidTypeDescriptor(field.type);
        }
        return;
    }

    throw new Error('Invalid type descriptor');
}

/**
 * Create a struct schema from field definitions.
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
    const structFields: StructField[] = [];

    for (const [fieldName, fieldType] of Object.entries(fields)) {
        assertValidTypeDescriptor(fieldType);
        structFields.push({
            name: fieldName,
            type: fieldType
        });
    }

    return {
        kind: 'struct',
        fields: structFields,
        name
    };
}

/**
 * Create a fixed-size array schema.
 *
 * @example
 * const positions = array("vec3f", 100);
 * const nested = array(array("f32", 4), 8);
 */
export function array(elementType: TypeDescriptor, count: number): ArraySchema {
    if (count <= 0 || !Number.isInteger(count)) {
        throw new Error(
            `Array count must be a positive integer, got: ${count}`
        );
    }

    assertValidTypeDescriptor(elementType);

    return {
        kind: 'array',
        elementType,
        count
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

import { type PrimitiveType, PRIMITIVE_INFO } from '../types/primitives.js';
import {
    type ArraySchema,
    type StructSchema,
    type TypeDescriptor
} from '../types/schema.js';

/**
 * Concrete packing rules supported by Volten's layout engine.
 *
 * These are intentionally more explicit than WGSL address spaces:
 * - `storage` means natural/storage-style layout
 * - `uniform-classic` means classic uniform rules without
 *   `uniform_buffer_standard_layout`
 */
export type LayoutRules = 'storage' | 'uniform-classic';

export interface LayoutField {
    readonly name: string;
    readonly type: TypeDescriptor;
    readonly offset: number;
    readonly layout: TypeLayout;
}

export interface PrimitiveLayout {
    readonly kind: 'primitive';
    readonly type: PrimitiveType;
    readonly size: number;
    readonly align: number;
    readonly requiredAlign: number;
}

export interface ArrayLayout {
    readonly kind: 'array';
    readonly size: number;
    readonly align: number;
    readonly requiredAlign: number;
    readonly stride: number;
    readonly count: number;
    readonly elementType: TypeDescriptor;
    readonly elementLayout: TypeLayout;
}

export interface StructLayout {
    readonly kind: 'struct';
    readonly size: number;
    readonly align: number;
    readonly requiredAlign: number;
    readonly fields: readonly LayoutField[];
}

export type TypeLayout = PrimitiveLayout | ArrayLayout | StructLayout;

/**
 * Round up to alignment boundary.
 */
export function roundUp(value: number, alignment: number): number {
    return Math.ceil(value / alignment) * alignment;
}

/**
 * Compute a concrete memory layout for a type descriptor under the specified
 * packing rules.
 *
 * For `uniform-classic` this applies WGSL classic-uniform constraints,
 * including:
 * - array element stride aligned to 16-byte boundaries
 * - extra spacing when a struct member has struct type
 */
export function computeTypeLayout(
    type: TypeDescriptor,
    layoutRules: LayoutRules
): TypeLayout {
    if (typeof type === 'string') {
        const info = PRIMITIVE_INFO[type];
        return {
            kind: 'primitive',
            type,
            size: info.size,
            align: info.alignment,
            requiredAlign: info.alignment
        };
    }

    if (type.kind === 'array') {
        return computeArrayLayout(type, layoutRules);
    }

    return computeStructLayout(type, layoutRules);
}

/**
 * Element stride for arrays of T under the chosen packing rules.
 * This is `roundUp(size(T), requiredAlign(T))`.
 */
export function elementStrideOf(
    type: TypeDescriptor,
    layoutRules: LayoutRules
): number {
    const layout = computeTypeLayout(type, layoutRules);
    return roundUp(layout.size, layout.requiredAlign);
}

function computeArrayLayout(
    type: ArraySchema,
    layoutRules: LayoutRules
): ArrayLayout {
    const elementLayout = computeTypeLayout(type.elementType, layoutRules);
    const naturalAlign = elementLayout.align;
    const requiredAlign =
        layoutRules === 'uniform-classic'
            ? roundUp(naturalAlign, 16)
            : naturalAlign;

    // Natural default stride in WGSL type layout.
    const naturalStride = roundUp(elementLayout.size, elementLayout.align);

    // Classic uniforms require array elements to start on 16-byte boundaries.
    const strideAlign =
        layoutRules === 'uniform-classic'
            ? Math.max(16, elementLayout.requiredAlign)
            : elementLayout.requiredAlign;
    const stride = roundUp(naturalStride, strideAlign);

    return {
        kind: 'array',
        size: stride * type.count,
        align: naturalAlign,
        requiredAlign,
        stride,
        count: type.count,
        elementType: type.elementType,
        elementLayout
    };
}

function computeStructLayout(
    type: StructSchema,
    layoutRules: LayoutRules
): StructLayout {
    const rawFields = type.fields.map((field) => {
        const layout = computeTypeLayout(field.type, layoutRules);
        return {
            name: field.name,
            type: field.type,
            layout
        };
    });

    let offset = 0;
    const fields: LayoutField[] = [];

    for (let i = 0; i < rawFields.length; i++) {
        const field = rawFields[i];
        offset = roundUp(offset, field.layout.requiredAlign);
        const fieldOffset = offset;
        offset += field.layout.size;

        // Classic-uniform rule: if this member has struct type, the next
        // member must start at least roundUp(16, SizeOf(member)) bytes later.
        if (
            layoutRules === 'uniform-classic' &&
            field.layout.kind === 'struct' &&
            i < rawFields.length - 1
        ) {
            const minAdvance = roundUp(field.layout.size, 16);
            offset = Math.max(offset, fieldOffset + minAdvance);
        }

        fields.push({
            name: field.name,
            type: field.type,
            offset: fieldOffset,
            layout: field.layout
        });
    }

    const naturalAlign =
        rawFields.length === 0
            ? 1
            : rawFields.reduce((max, f) => Math.max(max, f.layout.align), 1);
    const requiredAlign =
        layoutRules === 'uniform-classic'
            ? roundUp(naturalAlign, 16)
            : naturalAlign;
    const size = roundUp(offset, requiredAlign);

    return {
        kind: 'struct',
        size,
        align: naturalAlign,
        requiredAlign,
        fields
    };
}

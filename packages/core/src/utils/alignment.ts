import {
    type PrimitiveType,
    getPrimitiveInfo,
    PRIMITIVE_INFO
} from '../types/primitives.js';
import { type TypeDescriptor } from '../types/schema.js';
import {
    computeTypeLayout,
    elementStrideOf,
    roundUp as roundUpInternal,
    type LayoutRules,
    type TypeLayout
} from './layout.js';

export interface LayoutOptions {
    /**
     * Explicit packing rules to apply.
     *
     * `storage` is also the rule set used for uniforms when
     * `uniform_buffer_standard_layout` is enabled.
     */
    layoutRules?: LayoutRules;
}

function resolveLayoutRules(options?: LayoutOptions): LayoutRules {
    return options?.layoutRules ?? 'storage';
}

/**
 * Round up a value to the nearest multiple of alignment.
 */
export function roundUp(value: number, alignment: number): number {
    return roundUpInternal(value, alignment);
}

/**
 * Pack a scalar value into a DataView at the given offset.
 */
function packScalar(
    view: DataView,
    offset: number,
    value: number | boolean,
    type: PrimitiveType
): void {
    const info = PRIMITIVE_INFO[type];
    const numValue = typeof value === 'boolean' ? (value ? 1 : 0) : value;

    switch (info.baseType) {
        case 'f32':
            view.setFloat32(offset, numValue, true);
            break;
        case 'u32':
            view.setUint32(offset, numValue >>> 0, true);
            break;
        case 'i32':
            view.setInt32(offset, numValue | 0, true);
            break;
        case 'f16':
            // Kept as f32 for now, consistent with existing f16 handling in Volten.
            view.setFloat32(offset, numValue, true);
            break;
    }
}

/**
 * Pack a vector value into a DataView at the given offset.
 * Vector can be an array [x, y, z, w] or typed array.
 */
function packVector(
    view: DataView,
    offset: number,
    value: ArrayLike<number>,
    type: PrimitiveType
): void {
    const info = PRIMITIVE_INFO[type];
    const componentSize = info.size / info.components;

    for (let i = 0; i < info.components; i++) {
        const componentValue = i < value.length ? value[i] : 0;
        const componentOffset = offset + i * componentSize;

        switch (info.baseType) {
            case 'f32':
                view.setFloat32(componentOffset, componentValue, true);
                break;
            case 'u32':
                view.setUint32(componentOffset, componentValue >>> 0, true);
                break;
            case 'i32':
                view.setInt32(componentOffset, componentValue | 0, true);
                break;
            case 'f16':
                view.setFloat32(componentOffset, componentValue, true);
                break;
        }
    }
}

/**
 * Pack a matrix value into a DataView at the given offset.
 * Matrix is column-major: mat4x4f = 4 columns of vec4f.
 * Can be passed as flat array or nested array.
 */
function packMatrix(
    view: DataView,
    offset: number,
    value: ArrayLike<number> | ArrayLike<ArrayLike<number>>,
    type: PrimitiveType
): void {
    const match = type.match(/mat(\d)x(\d)(f|h)/);
    if (!match) {
        throw new Error(`Invalid matrix type: ${type}`);
    }

    const cols = parseInt(match[1], 10);
    const rows = parseInt(match[2], 10);

    const columnStride = rows <= 2 ? 8 : 16;

    let flatValues: number[];
    if (value.length > 0 && typeof value[0] === 'object') {
        flatValues = [];
        for (let c = 0; c < cols; c++) {
            const col = (value as ArrayLike<ArrayLike<number>>)[c] || [];
            for (let r = 0; r < rows; r++) {
                flatValues.push(col[r] ?? 0);
            }
        }
    } else {
        flatValues = Array.from(value as ArrayLike<number>);
    }

    for (let c = 0; c < cols; c++) {
        const columnOffset = offset + c * columnStride;
        for (let r = 0; r < rows; r++) {
            const valueIndex = c * rows + r;
            const val =
                valueIndex < flatValues.length ? flatValues[valueIndex] : 0;
            view.setFloat32(columnOffset + r * 4, val, true);
        }
    }
}

/**
 * Pack a primitive value.
 */
function packPrimitive(
    view: DataView,
    offset: number,
    value: number | boolean | ArrayLike<number>,
    type: PrimitiveType
): void {
    const info = PRIMITIVE_INFO[type];

    if (info.components === 1) {
        packScalar(view, offset, value as number | boolean, type);
    } else if (type.startsWith('mat')) {
        packMatrix(view, offset, value as ArrayLike<number>, type);
    } else {
        packVector(view, offset, value as ArrayLike<number>, type);
    }
}

function packValueWithLayout(
    view: DataView,
    offset: number,
    value: unknown,
    layout: TypeLayout
): void {
    if (layout.kind === 'primitive') {
        packPrimitive(
            view,
            offset,
            value as number | boolean | ArrayLike<number>,
            layout.type
        );
        return;
    }

    if (layout.kind === 'struct') {
        const record = value as Record<string, unknown>;
        for (const field of layout.fields) {
            const fieldValue = record[field.name];
            if (fieldValue === undefined) {
                continue;
            }
            packValueWithLayout(
                view,
                offset + field.offset,
                fieldValue,
                field.layout
            );
        }
        return;
    }

    const values = value as ArrayLike<unknown>;
    for (let i = 0; i < layout.count && i < values.length; i++) {
        packValueWithLayout(
            view,
            offset + i * layout.stride,
            values[i],
            layout.elementLayout
        );
    }
}

/**
 * Main packing function: pack JavaScript data into an ArrayBuffer.
 *
 * `data` is an array of elements, each encoded as one element of `type`.
 */
export function pack(
    data: ArrayLike<unknown>,
    type: TypeDescriptor,
    options?: LayoutOptions
): ArrayBuffer {
    const layoutRules = resolveLayoutRules(options);
    const elementLayout = computeTypeLayout(type, layoutRules);
    const elementStride = elementStrideOf(type, layoutRules);
    const byteLength = elementStride * data.length;

    const buffer = new ArrayBuffer(byteLength);
    const view = new DataView(buffer);

    for (let i = 0; i < data.length; i++) {
        packValueWithLayout(view, i * elementStride, data[i], elementLayout);
    }

    return buffer;
}

/**
 * Calculate stride (bytes per element) for arrays of this type.
 */
export function getStride(
    type: TypeDescriptor,
    options?: LayoutOptions
): number {
    return elementStrideOf(type, resolveLayoutRules(options));
}

/**
 * Calculate total byte length for an array of elements.
 */
export function getByteLength(
    count: number,
    type: TypeDescriptor,
    options?: LayoutOptions
): number {
    return getStride(type, options) * count;
}

/**
 * Get the appropriate TypedArray constructor for a descriptor.
 * Returns undefined for structs (which should be returned as raw ArrayBuffers).
 */
export function getTypedArrayForType(
    type: TypeDescriptor
):
    | Float32ArrayConstructor
    | Uint32ArrayConstructor
    | Int32ArrayConstructor
    | undefined {
    if (typeof type === 'string') {
        const info = getPrimitiveInfo(type);
        switch (info.baseType) {
            case 'f32':
            case 'f16':
                return Float32Array;
            case 'u32':
                return Uint32Array;
            case 'i32':
                return Int32Array;
        }
    }
    if (type.kind === 'array') {
        return getTypedArrayForType(type.elementType);
    }
    return undefined;
}

function unpackScalar(
    view: DataView,
    offset: number,
    type: PrimitiveType
): number | boolean {
    const info = getPrimitiveInfo(type);
    switch (info.baseType) {
        case 'f32':
            return view.getFloat32(offset, true);
        case 'u32':
            return type === 'bool'
                ? view.getUint32(offset, true) !== 0
                : view.getUint32(offset, true);
        case 'i32':
            return view.getInt32(offset, true);
        case 'f16':
            return view.getFloat32(offset, true);
    }
}

function unpackVector(
    view: DataView,
    offset: number,
    type: PrimitiveType
): number[] {
    const info = getPrimitiveInfo(type);
    const componentSize = info.size / info.components;
    const result: number[] = [];

    for (let i = 0; i < info.components; i++) {
        const componentOffset = offset + i * componentSize;
        switch (info.baseType) {
            case 'f32':
                result.push(view.getFloat32(componentOffset, true));
                break;
            case 'u32':
                result.push(view.getUint32(componentOffset, true));
                break;
            case 'i32':
                result.push(view.getInt32(componentOffset, true));
                break;
            case 'f16':
                result.push(view.getFloat32(componentOffset, true));
                break;
        }
    }

    return result;
}

function unpackMatrix(
    view: DataView,
    offset: number,
    type: PrimitiveType
): number[] {
    const match = type.match(/mat(\d)x(\d)(f|h)/);
    if (!match) {
        throw new Error(`Invalid matrix type: ${type}`);
    }

    const cols = parseInt(match[1], 10);
    const rows = parseInt(match[2], 10);
    const columnStride = rows <= 2 ? 8 : 16;

    const result: number[] = [];
    for (let c = 0; c < cols; c++) {
        const columnOffset = offset + c * columnStride;
        for (let r = 0; r < rows; r++) {
            result.push(view.getFloat32(columnOffset + r * 4, true));
        }
    }
    return result;
}

function unpackPrimitive(
    view: DataView,
    offset: number,
    type: PrimitiveType
): number | boolean | number[] {
    const info = getPrimitiveInfo(type);
    if (info.components === 1) return unpackScalar(view, offset, type);
    if (type.startsWith('mat')) return unpackMatrix(view, offset, type);
    return unpackVector(view, offset, type);
}

function unpackValueWithLayout(
    view: DataView,
    offset: number,
    layout: TypeLayout
): unknown {
    if (layout.kind === 'primitive') {
        return unpackPrimitive(view, offset, layout.type);
    }

    if (layout.kind === 'struct') {
        const out: Record<string, unknown> = {};
        for (const field of layout.fields) {
            out[field.name] = unpackValueWithLayout(
                view,
                offset + field.offset,
                field.layout
            );
        }
        return out;
    }

    const out: unknown[] = [];
    for (let i = 0; i < layout.count; i++) {
        out.push(
            unpackValueWithLayout(
                view,
                offset + i * layout.stride,
                layout.elementLayout
            )
        );
    }
    return out;
}

/**
 * Inverse of pack: decode a mapped GPU buffer or any packed ArrayBuffer back
 * into structured JavaScript values using the same layout rules.
 */
export function unpack(
    buffer: ArrayBuffer,
    type: TypeDescriptor,
    options?: LayoutOptions
): unknown[] {
    const layoutRules = resolveLayoutRules(options);
    const layout = computeTypeLayout(type, layoutRules);
    const stride = elementStrideOf(type, layoutRules);
    const count = Math.floor(buffer.byteLength / stride);

    const view = new DataView(buffer);
    const result: unknown[] = [];

    for (let i = 0; i < count; i++) {
        result.push(unpackValueWithLayout(view, i * stride, layout));
    }

    return result;
}

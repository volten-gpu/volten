import {
    type PrimitiveType,
    type TypeInfo,
    isPrimitiveType,
    getPrimitiveInfo,
    PRIMITIVE_INFO,
} from '../types/primitives.js';
import {
    type TypeDescriptor,
    type StructSchema,
    type ArraySchema,
    resolveType,
} from '../types/schema.js';

/**
 * Round up a value to the nearest multiple of alignment
 */
export function roundUp(value: number, alignment: number): number {
    return Math.ceil(value / alignment) * alignment;
}

/**
 * Get the appropriate TypedArray constructor for a base type
 */
function getTypedArrayConstructor(
    baseType: 'f32' | 'u32' | 'i32' | 'f16'
): Float32ArrayConstructor | Uint32ArrayConstructor | Int32ArrayConstructor {
    switch (baseType) {
        case 'f32':
            return Float32Array;
        case 'u32':
            return Uint32Array;
        case 'i32':
            return Int32Array;
        case 'f16':
            // f16 is stored as f32 for now (we'd need Float16Array which isn't standard)
            return Float32Array;
    }
}

/**
 * Pack a scalar value into a DataView at the given offset
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
            view.setFloat32(offset, numValue, true); // little-endian
            break;
        case 'u32':
            view.setUint32(offset, numValue >>> 0, true);
            break;
        case 'i32':
            view.setInt32(offset, numValue | 0, true);
            break;
        case 'f16':
            // Store as f32 for now (proper f16 would need conversion)
            view.setFloat32(offset, numValue, true);
            break;
    }
}

/**
 * Pack a vector value into a DataView at the given offset
 * Vector can be an array [x, y, z, w] or typed array
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
 * Pack a matrix value into a DataView at the given offset
 * Matrix is column-major: mat4x4f = 4 columns of vec4f
 * Can be passed as flat array or nested array
 */
function packMatrix(
    view: DataView,
    offset: number,
    value: ArrayLike<number> | ArrayLike<ArrayLike<number>>,
    type: PrimitiveType
): void {
    const info = PRIMITIVE_INFO[type];

    // Parse matrix dimensions from type name (e.g., "mat4x3f" -> cols=4, rows=3)
    const match = type.match(/mat(\d)x(\d)f/);
    if (!match) {
        throw new Error(`Invalid matrix type: ${type}`);
    }

    const cols = parseInt(match[1], 10);
    const rows = parseInt(match[2], 10);

    // Determine column stride based on row count
    // vec2 = 8 bytes, vec3/vec4 = 16 bytes (vec3 padded to 16)
    const columnStride = rows <= 2 ? 8 : 16;

    // Flatten if nested array
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

    // Pack column by column
    for (let c = 0; c < cols; c++) {
        const columnOffset = offset + c * columnStride;
        for (let r = 0; r < rows; r++) {
            const valueIndex = c * rows + r;
            const val = valueIndex < flatValues.length ? flatValues[valueIndex] : 0;
            view.setFloat32(columnOffset + r * 4, val, true);
        }
    }
}

/**
 * Pack a single primitive value
 */
function packPrimitive(
    view: DataView,
    offset: number,
    value: number | boolean | ArrayLike<number>,
    type: PrimitiveType
): void {
    const info = PRIMITIVE_INFO[type];

    if (info.components === 1) {
        // Scalar
        packScalar(view, offset, value as number | boolean, type);
    } else if (type.startsWith('mat')) {
        // Matrix
        packMatrix(view, offset, value as ArrayLike<number>, type);
    } else {
        // Vector
        packVector(view, offset, value as ArrayLike<number>, type);
    }
}

/**
 * Pack a struct value according to its schema
 */
function packStructValue(
    view: DataView,
    offset: number,
    value: Record<string, unknown>,
    schema: StructSchema
): void {
    for (const field of schema.fields) {
        const fieldValue = value[field.name];
        if (fieldValue === undefined) {
            // Leave as zeros (already initialized)
            continue;
        }
        packValue(view, offset + field.offset, fieldValue, field.type);
    }
}

/**
 * Pack an array of values according to its schema
 */
function packArrayValue(
    view: DataView,
    offset: number,
    values: ArrayLike<unknown>,
    schema: ArraySchema
): void {
    for (let i = 0; i < schema.count && i < values.length; i++) {
        packValue(view, offset + i * schema.stride, values[i], schema.elementType);
    }
}

/**
 * Pack any value according to its type descriptor
 */
function packValue(
    view: DataView,
    offset: number,
    value: unknown,
    type: TypeDescriptor
): void {
    if (typeof type === 'string') {
        packPrimitive(view, offset, value as number | boolean | ArrayLike<number>, type);
    } else if (type.kind === 'struct') {
        packStructValue(view, offset, value as Record<string, unknown>, type);
    } else if (type.kind === 'array') {
        packArrayValue(view, offset, value as ArrayLike<unknown>, type);
    }
}

/**
 * Main packing function: pack JavaScript data into an ArrayBuffer
 * 
 * @param data - Array of values to pack (each becomes one element)
 * @param type - Type descriptor (primitive string, struct, or array schema)
 * @returns Packed ArrayBuffer ready for GPU upload
 * 
 * @example
 * // Pack array of floats
 * const floats = pack([1, 2, 3], "f32");
 * 
 * // Pack array of vec3f (note: stride is 16, not 12!)
 * const vecs = pack([[1, 2, 3], [4, 5, 6]], "vec3f");
 * 
 * // Pack array of structs
 * const Particle = struct("Particle", { position: "vec3f", mass: "f32" });
 * const particles = pack([
 *   { position: [0, 1, 2], mass: 1.0 },
 *   { position: [3, 4, 5], mass: 2.0 }
 * ], Particle);
 */
export function pack(data: ArrayLike<unknown>, type: TypeDescriptor): ArrayBuffer {
    const resolved = resolveType(type);
    const stride = roundUp(resolved.size, resolved.alignment);
    const byteLength = stride * data.length;

    const buffer = new ArrayBuffer(byteLength);
    const view = new DataView(buffer);

    for (let i = 0; i < data.length; i++) {
        packValue(view, i * stride, data[i], type);
    }

    return buffer;
}

/**
 * Calculate the stride (bytes per element) for a type
 */
export function getStride(type: TypeDescriptor): number {
    const resolved = resolveType(type);
    return roundUp(resolved.size, resolved.alignment);
}

/**
 * Calculate total byte length for an array of elements
 */
export function getByteLength(count: number, type: TypeDescriptor): number {
    return getStride(type) * count;
}

/**
 * Get the appropriate TypedArray constructor for a given descriptor.
 * Returns undefined for structs (which should be returned as raw ArrayBuffers).
 */
export function getTypedArrayForType(
    type: TypeDescriptor
): Float32ArrayConstructor | Uint32ArrayConstructor | Int32ArrayConstructor | undefined {
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
    return undefined; // Struct
}

/**
 * Unpack a scalar value from a DataView at the given offset
 */
function unpackScalar(view: DataView, offset: number, type: PrimitiveType): number | boolean {
    const info = getPrimitiveInfo(type);
    switch (info.baseType) {
        case 'f32': return view.getFloat32(offset, true);
        case 'u32': return type === 'bool' ? (view.getUint32(offset, true) !== 0) : view.getUint32(offset, true);
        case 'i32': return view.getInt32(offset, true);
        case 'f16': return view.getFloat32(offset, true);
    }
}

/**
 * Unpack a vector value from a DataView at the given offset
 */
function unpackVector(view: DataView, offset: number, type: PrimitiveType): number[] {
    const info = getPrimitiveInfo(type);
    const componentSize = info.size / info.components;
    const result: number[] = [];

    for (let i = 0; i < info.components; i++) {
        const componentOffset = offset + i * componentSize;
        switch (info.baseType) {
            case 'f32': result.push(view.getFloat32(componentOffset, true)); break;
            case 'u32': result.push(view.getUint32(componentOffset, true)); break;
            case 'i32': result.push(view.getInt32(componentOffset, true)); break;
            case 'f16': result.push(view.getFloat32(componentOffset, true)); break;
        }
    }
    return result;
}

/**
 * Unpack a matrix value from a DataView at the given offset
 */
function unpackMatrix(view: DataView, offset: number, type: PrimitiveType): number[] {
    const match = type.match(/mat(\d)x(\d)f/);
    if (!match) throw new Error(`Invalid matrix type: ${type}`);

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

/**
 * Unpack a single primitive value
 */
function unpackPrimitive(view: DataView, offset: number, type: PrimitiveType): number | boolean | number[] {
    const info = getPrimitiveInfo(type);
    if (info.components === 1) return unpackScalar(view, offset, type);
    if (type.startsWith('mat')) return unpackMatrix(view, offset, type);
    return unpackVector(view, offset, type);
}

/**
 * Unpack a struct value according to its schema
 */
function unpackStructValue(view: DataView, offset: number, schema: StructSchema): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    for (const field of schema.fields) {
        result[field.name] = unpackValue(view, offset + field.offset, field.type);
    }
    return result;
}

/**
 * Unpack an array of values according to its schema
 */
function unpackArrayValue(view: DataView, offset: number, schema: ArraySchema): unknown[] {
    const result: unknown[] = [];
    for (let i = 0; i < schema.count; i++) {
        result.push(unpackValue(view, offset + i * schema.stride, schema.elementType));
    }
    return result;
}

/**
 * Unpack any value according to its type descriptor
 */
function unpackValue(view: DataView, offset: number, type: TypeDescriptor): unknown {
    if (typeof type === 'string') return unpackPrimitive(view, offset, type as PrimitiveType);
    if (type.kind === 'struct') return unpackStructValue(view, offset, type);
    if (type.kind === 'array') return unpackArrayValue(view, offset, type);
    throw new Error('Invalid type descriptor');
}

/**
 * Inverse of pack: unpacks an ArrayBuffer into an array of JavaScript objects/arrays/primitives
 *
 * @param buffer - The packed ArrayBuffer from the GPU
 * @param type - Type descriptor (primitive string, struct, or array schema)
 * @returns Array of unpacked values
 */
export function unpack(buffer: ArrayBuffer, type: TypeDescriptor): unknown[] {
    const resolved = resolveType(type);
    const stride = roundUp(resolved.size, resolved.alignment);

    const count = Math.floor(buffer.byteLength / stride);
    const view = new DataView(buffer);
    const result: unknown[] = [];

    for (let i = 0; i < count; i++) {
        result.push(unpackValue(view, i * stride, type));
    }

    return result;
}

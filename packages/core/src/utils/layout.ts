import { type PrimitiveType, PRIMITIVE_INFO } from "../types/primitives.js";
import {
  type ArraySchema,
  type StructSchema,
  type TypeDescriptor,
} from "../types/schema.js";

export type AddressSpace = "storage" | "uniform";

export interface LayoutField {
  readonly name: string;
  readonly type: TypeDescriptor;
  readonly offset: number;
  readonly layout: TypeLayout;
}

export interface PrimitiveLayout {
  readonly kind: "primitive";
  readonly type: PrimitiveType;
  readonly size: number;
  readonly align: number;
  readonly requiredAlign: number;
}

export interface ArrayLayout {
  readonly kind: "array";
  readonly size: number;
  readonly align: number;
  readonly requiredAlign: number;
  readonly stride: number;
  readonly count: number;
  readonly elementType: TypeDescriptor;
  readonly elementLayout: TypeLayout;
}

export interface StructLayout {
  readonly kind: "struct";
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
 * Compute a concrete memory layout for a type descriptor in the specified address space.
 *
 * For `uniform` this applies WGSL address-space constraints (no
 * uniform_buffer_standard_layout extension assumed), including:
 * - array element stride aligned to 16-byte boundaries
 * - extra spacing when a struct member has struct type
 */
export function computeTypeLayout(
  type: TypeDescriptor,
  addressSpace: AddressSpace,
): TypeLayout {
  if (typeof type === "string") {
    const info = PRIMITIVE_INFO[type];
    return {
      kind: "primitive",
      type,
      size: info.size,
      align: info.alignment,
      requiredAlign: info.alignment,
    };
  }

  if (type.kind === "array") {
    return computeArrayLayout(type, addressSpace);
  }

  return computeStructLayout(type, addressSpace);
}

/**
 * RequiredAlignOf(T, C) for supported host-shareable descriptors.
 */
export function requiredAlignOf(
  type: TypeDescriptor,
  addressSpace: AddressSpace,
): number {
  const layout = computeTypeLayout(type, addressSpace);
  return layout.requiredAlign;
}

/**
 * Element stride for arrays of T in address space C.
 * This is `roundUp(size(T), RequiredAlignOf(T, C))`.
 */
export function elementStrideOf(
  type: TypeDescriptor,
  addressSpace: AddressSpace,
): number {
  const layout = computeTypeLayout(type, addressSpace);
  return roundUp(layout.size, layout.requiredAlign);
}

function computeArrayLayout(
  type: ArraySchema,
  addressSpace: AddressSpace,
): ArrayLayout {
  const elementLayout = computeTypeLayout(type.elementType, addressSpace);
  const naturalAlign = elementLayout.align;
  const requiredAlign =
    addressSpace === "uniform" ? roundUp(naturalAlign, 16) : naturalAlign;

  // Natural default stride in WGSL type layout
  const naturalStride = roundUp(elementLayout.size, elementLayout.align);

  // Uniform (without uniform_buffer_standard_layout): array elements are aligned to 16 bytes.
  const strideAlign =
    addressSpace === "uniform"
      ? Math.max(16, elementLayout.requiredAlign)
      : elementLayout.requiredAlign;
  const stride = roundUp(naturalStride, strideAlign);

  return {
    kind: "array",
    size: stride * type.count,
    align: naturalAlign,
    requiredAlign,
    stride,
    count: type.count,
    elementType: type.elementType,
    elementLayout,
  };
}

function computeStructLayout(
  type: StructSchema,
  addressSpace: AddressSpace,
): StructLayout {
  const rawFields = type.fields.map((field) => {
    const layout = computeTypeLayout(field.type, addressSpace);
    return {
      name: field.name,
      type: field.type,
      layout,
    };
  });

  let offset = 0;
  const fields: LayoutField[] = [];

  for (let i = 0; i < rawFields.length; i++) {
    const field = rawFields[i];
    offset = roundUp(offset, field.layout.requiredAlign);
    const fieldOffset = offset;
    offset += field.layout.size;

    // Uniform layout rule: if this member has struct type, next member start
    // must be at least roundUp(16, SizeOf(structMemberType)) bytes from this start.
    if (
      addressSpace === "uniform" &&
      field.layout.kind === "struct" &&
      i < rawFields.length - 1
    ) {
      const minAdvance = roundUp(field.layout.size, 16);
      offset = Math.max(offset, fieldOffset + minAdvance);
    }

    fields.push({
      name: field.name,
      type: field.type,
      offset: fieldOffset,
      layout: field.layout,
    });
  }

  const naturalAlign =
    rawFields.length === 0
      ? 1
      : rawFields.reduce((max, f) => Math.max(max, f.layout.align), 1);
  const requiredAlign =
    addressSpace === "uniform" ? roundUp(naturalAlign, 16) : naturalAlign;
  const size = roundUp(offset, requiredAlign);

  return {
    kind: "struct",
    size,
    align: naturalAlign,
    requiredAlign,
    fields,
  };
}

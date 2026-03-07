import {
    type ArraySchema,
    getWgslType,
    type StructSchema,
    type TypeDescriptor
} from '../types/schema.js';
import {
    computeTypeLayout,
    elementStrideOf,
    type StructLayout
} from '../utils/layout.js';
import type { UniformLayoutMode } from '../utils/uniform-layout.js';

export interface BindingTypeInfo {
    readonly name: string;
    readonly wgslAddressSpace: 'storage' | 'uniform';
    readonly typeDescriptor?: TypeDescriptor;
}

type StructVariant = 'natural' | 'uniform-classic';

interface StructVariantRecord {
    readonly schema: StructSchema;
    variant: StructVariant;
}

/**
 * Builds WGSL struct declarations required by the provided binding type descriptors.
 *
 * Natural variant is used for:
 * - storage bindings
 * - uniform bindings with standard layout mode
 *
 * Classic variant is used for:
 * - uniform bindings with classic layout mode
 */
export function generateTypeDeclarations(
    bindings: readonly BindingTypeInfo[],
    uniformLayoutMode: UniformLayoutMode
): string {
    const structsByName = new Map<string, StructVariantRecord>();

    for (const binding of bindings) {
        if (!binding.typeDescriptor) {
            continue;
        }

        const variant: StructVariant =
            binding.wgslAddressSpace === 'uniform' &&
            uniformLayoutMode === 'classic'
                ? 'uniform-classic'
                : 'natural';

        visitTypeDescriptor(
            binding.typeDescriptor,
            variant,
            binding.name,
            binding.name,
            structsByName
        );
    }

    if (structsByName.size === 0) {
        return '';
    }

    const emitted = new Set<string>();
    const visiting = new Set<string>();
    const declarations: string[] = [];

    for (const name of structsByName.keys()) {
        emitStruct(name, structsByName, visiting, emitted, declarations);
    }

    return declarations.join('\n\n');
}

function visitTypeDescriptor(
    type: TypeDescriptor,
    variant: StructVariant,
    bindingName: string,
    path: string,
    structsByName: Map<string, StructVariantRecord>
): void {
    if (typeof type === 'string') {
        return;
    }

    if (type.kind === 'array') {
        validateClassicUniformArray(type, variant, bindingName, path);
        visitTypeDescriptor(
            type.elementType,
            variant,
            bindingName,
            `${path}[]`,
            structsByName
        );
        return;
    }

    // if it's not a string nor an array, it's a struct

    registerStructVariant(type, variant, bindingName, path, structsByName);
    for (const field of type.fields) {
        visitTypeDescriptor(
            field.type,
            variant,
            bindingName,
            `${path}.${field.name}`,
            structsByName
        );
    }
}

function registerStructVariant(
    schema: StructSchema,
    variant: StructVariant,
    bindingName: string,
    path: string,
    structsByName: Map<string, StructVariantRecord>
): void {
    const existing = structsByName.get(schema.name);
    if (!existing) {
        structsByName.set(schema.name, { schema, variant });
        return;
    }

    if (existing.schema !== schema) {
        throw new Error(
            `Volten Error: Struct name collision for "${schema.name}".\n` +
                `  Found multiple schema objects with the same name while processing binding "${bindingName}" at "${path}".\n` +
                '  Hint: use unique struct names to avoid ambiguous WGSL declarations.'
        );
    }

    if (existing.variant !== variant) {
        throw new Error(
            `Volten Error: Struct "${schema.name}" is used by both storage and classic-uniform layouts in one shader.\n` +
                `  Binding "${bindingName}" at "${path}" requires ${variant}, but previous usage requires ${existing.variant}.\n` +
                '  Hint: use separate schema names for storage vs uniform, or run with uniform_buffer_standard_layout.'
        );
    }
}

function validateClassicUniformArray(
    arrayType: ArraySchema,
    variant: StructVariant,
    bindingName: string,
    path: string
): void {
    if (variant !== 'uniform-classic') {
        return;
    }

    const naturalStride = elementStrideOf(arrayType.elementType, 'storage');
    if (naturalStride % 16 === 0) {
        return;
    }

    throw new Error(
        `Volten Error: Uniform binding "${bindingName}" contains "${path}" with element stride ${naturalStride} bytes.\n` +
            '  Classic uniform layout requires 16-byte array stride, which would need wrapper element types.\n' +
            '  Volten cannot auto-generate those wrappers without changing field-access semantics.\n' +
            '  Hint: enable uniform_buffer_standard_layout (auto/standard mode), redesign the uniform type, or use RawBuffer.'
    );
}

function emitStruct(
    name: string,
    structsByName: Map<string, StructVariantRecord>,
    visiting: Set<string>,
    emitted: Set<string>,
    declarations: string[]
): void {
    if (emitted.has(name)) {
        return;
    }

    if (visiting.has(name)) {
        throw new Error(
            `Volten Error: Recursive struct dependency detected while emitting "${name}".`
        );
    }

    const record = structsByName.get(name);
    if (!record) {
        return;
    }

    visiting.add(name);
    for (const depName of collectStructDependencies(record.schema)) {
        emitStruct(depName, structsByName, visiting, emitted, declarations);
    }
    visiting.delete(name);

    declarations.push(
        record.variant === 'natural'
            ? renderNaturalStruct(record.schema)
            : renderClassicUniformStruct(record.schema)
    );
    emitted.add(name);
}

function collectStructDependencies(schema: StructSchema): string[] {
    const deps = new Set<string>();
    for (const field of schema.fields) {
        collectStructDependenciesFromType(field.type, deps);
    }
    return [...deps];
}

function collectStructDependenciesFromType(
    type: TypeDescriptor,
    deps: Set<string>
): void {
    if (typeof type === 'string') {
        return;
    }

    if (type.kind === 'array') {
        collectStructDependenciesFromType(type.elementType, deps);
        return;
    }

    deps.add(type.name);
    for (const field of type.fields) {
        collectStructDependenciesFromType(field.type, deps);
    }
}

function renderNaturalStruct(schema: StructSchema): string {
    const lines: string[] = [];
    lines.push(`struct ${schema.name} {`);
    for (const field of schema.fields) {
        lines.push(`    ${field.name}: ${getWgslType(field.type)},`);
    }
    lines.push('};');
    return lines.join('\n');
}

function renderClassicUniformStruct(schema: StructSchema): string {
    const uniformLayout = computeTypeLayout(
        schema,
        'uniform-classic'
    ) as StructLayout;
    const naturalLayout = computeTypeLayout(schema, 'storage') as StructLayout;

    const naturalOffsetsByName = new Map<string, number>();
    for (const field of naturalLayout.fields) {
        naturalOffsetsByName.set(field.name, field.offset);
    }

    const lines: string[] = [];
    lines.push(`struct ${schema.name} {`);
    for (const field of schema.fields) {
        const uniformField = uniformLayout.fields.find(
            (f) => f.name === field.name
        );
        if (!uniformField) {
            throw new Error(
                `Volten Error: Failed to resolve uniform layout field "${field.name}" in struct "${schema.name}".`
            );
        }

        const naturalOffset = naturalOffsetsByName.get(field.name) ?? 0;
        const needsAlign16 = uniformField.offset !== naturalOffset;
        const prefix = needsAlign16 ? '@align(16) ' : '';
        lines.push(`    ${prefix}${field.name}: ${getWgslType(field.type)},`);
    }
    lines.push('};');
    return lines.join('\n');
}

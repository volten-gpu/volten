// Binding generation for compute shaders
// Classifies user-provided bindings and generates WGSL declarations

import { Buffer } from '../data/buffer.js';
import { RawBuffer } from '../data/raw-buffer.js';
import { Uniform } from '../data/uniform.js';
import { resolveConcreteBuffer } from '../graph/compiler.js';
import { type Handle, isHandle } from '../graph/node.js';
import type { TypeDescriptor } from '../types/schema.js';
import {
    generateTypeDeclarations,
    type BindingTypeInfo
} from './type-declarations.js';
import type { Kernel } from './kernel.js';
import {
    type UniformLayoutMode,
    UNIFORM_BUFFER_STANDARD_LAYOUT_EXTENSION
} from '../utils/uniform-layout.js';

/**
 * A classified binding entry, ready for WGSL generation and GPU bind group creation.
 */
export interface BindingEntry {
    /** Binding index (@binding(N)) */
    readonly index: number;
    /** Name used in the shader (the key from the user's bindings record) */
    readonly name: string;
    /** WGSL type string (e.g. "array<f32>") */
    readonly wgslType: string;
    /** WGSL address space ("storage" or "uniform") */
    readonly wgslAddressSpace: 'storage' | 'uniform';
    /** WGSL access qualifier ("read" or "read_write"), storage-only */
    readonly wgslAccess?: string;
    /** The source object: Buffer, RawBuffer, Uniform, or Handle */
    readonly source: Buffer | RawBuffer | Uniform | Handle;
    /** Whether this is a Handle from a previous node */
    readonly isHandle: boolean;
    /** Optional Volten type descriptor when available (Buffer/Uniform chains) */
    readonly typeDescriptor?: TypeDescriptor;
}

/**
 * Check if a value is a Buffer instance.
 */
function isBuffer(value: unknown): value is Buffer {
    return value instanceof Buffer;
}

/**
 * Check if a value is a RawBuffer instance.
 */
function isRawBuffer(value: unknown): value is RawBuffer {
    return value instanceof RawBuffer;
}

/**
 * Check if a value is a Uniform instance.
 */
function isUniform(value: unknown): value is Uniform {
    return value instanceof Uniform;
}

/**
 * Resolve the wgslType and wgslAccess from a Handle by walking up the Handle chain.
 * Recursively follows Handle → source node's binding until a concrete Buffer/RawBuffer is found.
 */
function resolveHandleSource(handle: Handle): {
    wgslType: string;
    wgslAddressSpace: 'storage' | 'uniform';
    wgslAccess?: string;
    typeDescriptor?: TypeDescriptor;
} {
    const source = handle._node._bindings[handle._name];
    if (isBuffer(source)) {
        return {
            wgslType: source.wgslType,
            wgslAddressSpace: 'storage',
            wgslAccess: source.wgslAccess,
            typeDescriptor: source.type
        };
    }
    if (isRawBuffer(source)) {
        return {
            wgslType: source.wgslType,
            wgslAddressSpace: 'storage',
            wgslAccess: source.wgslAccess
        };
    }
    if (isUniform(source)) {
        return {
            wgslType: source.wgslType,
            wgslAddressSpace: 'uniform',
            typeDescriptor: source.type
        };
    }
    if (isHandle(source)) {
        return resolveHandleSource(source as Handle);
    }
    throw new Error(
        `Volten Error: Handle "${handle._name}" does not resolve to a Buffer, RawBuffer, or Uniform.`
    );
}

/**
 * Generate binding entries from user-provided bindings and kernel definition.
 *
 * Classification rules:
 * - Buffer → var<storage, read|read_write> based on buffer.access
 * - RawBuffer → var<storage, read|read_write> based on buffer.access
 * - Handle → resolved from source node's binding (walks up the Handle chain)
 * - Anything else → throws an error
 *
 * @param bindings - User-provided bindings record
 * @param kernel - The kernel definition
 * @returns Array of classified binding entries
 */
export function generateBindings(
    bindings: Record<string, unknown>,
    kernel: Kernel,
    options?: { uniformLayoutMode?: UniformLayoutMode }
): BindingEntry[] {
    const entries: BindingEntry[] = [];
    let index = 0;
    const uniformLayoutMode = options?.uniformLayoutMode ?? 'classic';

    for (const [name, value] of Object.entries(bindings)) {
        if (isBuffer(value)) {
            entries.push({
                index: index++,
                name,
                wgslType: value.wgslType,
                wgslAddressSpace: 'storage',
                wgslAccess: value.wgslAccess,
                source: value,
                isHandle: false,
                typeDescriptor: value.type
            });
        } else if (isRawBuffer(value)) {
            entries.push({
                index: index++,
                name,
                wgslType: value.wgslType,
                wgslAddressSpace: 'storage',
                wgslAccess: value.wgslAccess,
                source: value,
                isHandle: false
            });
        } else if (isUniform(value)) {
            value.setLayoutMode(uniformLayoutMode);
            entries.push({
                index: index++,
                name,
                wgslType: value.wgslType,
                wgslAddressSpace: 'uniform',
                source: value,
                isHandle: false,
                typeDescriptor: value.type
            });
        } else if (isHandle(value)) {
            // Resolve actual type by walking up the Handle chain
            const resolved = resolveHandleSource(value);
            entries.push({
                index: index++,
                name,
                wgslType: resolved.wgslType,
                wgslAddressSpace: resolved.wgslAddressSpace,
                wgslAccess: resolved.wgslAccess,
                source: value,
                isHandle: true,
                typeDescriptor: resolved.typeDescriptor
            });
        } else if (typeof value === 'number') {
            throw new Error(
                `Volten Error: Binding "${name}" is a number (${value}). ` +
                    `Raw scalar bindings are not supported in v0.\n` +
                    `  Hint: Wrap it in a Buffer: new Buffer([${value}], "f32")`
            );
        } else {
            throw new Error(
                `Volten Error: Binding "${name}" has unsupported type: ${typeof value}.\n` +
                    `  Expected: Buffer, RawBuffer, Uniform, or Handle (output from a previous v.pass()).`
            );
        }
    }

    return entries;
}

/**
 * Generate WGSL binding declarations from binding entries.
 *
 * @example
 * // Input entry: { index: 0, name: "input", wgslType: "array<f32>", wgslAccess: "read" }
 * // Output: "@group(0) @binding(0) var<storage, read> input: array<f32>;"
 *
 * @param entries - Classified binding entries
 * @returns WGSL source string with binding declarations
 */
export function generateBindingWgsl(entries: BindingEntry[]): string {
    if (entries.length === 0) return '';

    return entries
        .map((entry) => {
            if (entry.wgslAddressSpace === 'uniform') {
                return `@group(0) @binding(${entry.index}) var<uniform> ${entry.name}: ${entry.wgslType};`;
            }
            return `@group(0) @binding(${entry.index}) var<storage, ${entry.wgslAccess}> ${entry.name}: ${entry.wgslType};`;
        })
        .join('\n');
}

/**
 * Assemble the full shader source: binding declarations + kernel source.
 *
 * The binding WGSL is prepended to the kernel's assembled source (which already
 * has builtin shorthands expanded and @compute/@workgroup_size injected).
 *
 * `assembleFullShader` is thus responsible for pass-time composition:
 * type declarations, binding declarations, optional extension lines.
 *
 * @param kernel - The kernel definition
 * @param entries - Classified binding entries
 * @returns Complete WGSL shader source ready for pipeline creation
 */
export function assembleFullShader(
    kernel: Kernel,
    entries: BindingEntry[],
    options?: { uniformLayoutMode?: UniformLayoutMode }
): string {
    const uniformLayoutMode = options?.uniformLayoutMode ?? 'classic';
    const bindingTypeInfo: BindingTypeInfo[] = entries.map((entry) => ({
        name: entry.name,
        wgslAddressSpace: entry.wgslAddressSpace,
        typeDescriptor: entry.typeDescriptor
    }));
    const typeDeclarations = generateTypeDeclarations(
        bindingTypeInfo,
        uniformLayoutMode
    );
    const bindingWgsl = generateBindingWgsl(entries);
    const requiresUniformStandardLayout =
        uniformLayoutMode === 'standard' &&
        entries.some((entry) => entry.wgslAddressSpace === 'uniform');
    const kernelSource = kernel.assembledSource;

    if (!bindingWgsl && !typeDeclarations && !requiresUniformStandardLayout) {
        return kernelSource;
    }

    const sections: string[] = [];
    if (requiresUniformStandardLayout) {
        sections.push(`requires ${UNIFORM_BUFFER_STANDARD_LAYOUT_EXTENSION};`);
    }
    if (typeDeclarations) {
        sections.push(typeDeclarations);
    }
    if (bindingWgsl) {
        sections.push(bindingWgsl);
    }
    sections.push(kernelSource);
    return sections.join('\n\n');
}

/**
 * Resolve thread dispatch dimensions from kernel config and bindings.
 *
 * Resolution order:
 * 1. Pass-time override (options.threads)
 * 2. Kernel-level threads spec
 * 3. Auto-inference from single buffer binding
 *
 * @param kernel - The kernel definition
 * @param bindings - User-provided bindings record
 * @param passThreads - Optional pass-time thread override
 * @returns Dispatch dimensions as [x, y, z]
 */
export function resolveBounds(
    kernel: Kernel,
    bindings: Record<string, unknown>,
    passThreads?:
        | number
        | [number]
        | [number, number]
        | [number, number, number]
): [number, number, number] {
    if (passThreads !== undefined) {
        if (typeof passThreads === 'number') {
            return [passThreads, 1, 1];
        }
        return normalizeThreads(passThreads);
    }

    const threads = kernel.threads;
    if (threads !== undefined) {
        if (typeof threads === 'number') {
            return [threads, 1, 1];
        }
        if (typeof threads === 'string') {
            const binding = bindings[threads];
            if (!binding) {
                throw new Error(
                    `Volten Error: Kernel threads spec references input "${threads}", ` +
                        `but it was not found in bindings.\n` +
                        `  Available bindings: ${Object.keys(bindings).join(', ')}`
                );
            }
            const count = getBindingCount(binding, threads);
            return [count, 1, 1];
        }
        if (typeof threads === 'function') {
            const result = threads(bindings);
            if (typeof result === 'number') {
                return [result, 1, 1];
            }
            return normalizeThreads(result);
        }
    }

    const bufferBindings = Object.entries(bindings).filter(([, v]) => {
        if (resolveConcreteBuffer(v)) {
            return true;
        }
        return false;
    });

    if (bufferBindings.length === 0) {
        throw new Error(
            'Volten Error: Cannot auto-infer thread count — no buffer bindings found.\n' +
                '  Hint: Specify threads explicitly: new Kernel(`...`, { threads: 1024 })'
        );
    }

    if (bufferBindings.length === 1) {
        const [name, buf] = bufferBindings[0];
        const count = getBindingCount(buf, name);
        return [count, 1, 1];
    }

    const outputNames = new Set(kernel.outputNames);
    const inputBuffers = bufferBindings.filter(
        ([name]) => !outputNames.has(name)
    );

    if (inputBuffers.length === 1) {
        const [name, buf] = inputBuffers[0];
        const count = getBindingCount(buf, name);
        return [count, 1, 1];
    }

    throw new Error(
        `Volten Error: Cannot auto-infer thread count — ${inputBuffers.length} input buffers found.\n` +
            `  Bindings: ${inputBuffers.map(([n]) => n).join(', ')}\n` +
            `  Hint: Specify threads explicitly: new Kernel(\`...\`, { threads: '${inputBuffers[0]?.[0] ?? 'input'}' })`
    );
}

export function resolveDispatch(
    kernel: Kernel,
    bindings: Record<string, unknown>,
    passThreads?:
        | number
        | [number]
        | [number, number]
        | [number, number, number]
): [number, number, number] {
    const bounds = resolveBounds(kernel, bindings, passThreads);
    return threadsToDispatch3D(bounds, kernel.workgroupSize);
}

/**
 * Get the element count from a binding value.
 * Handles are resolved by walking up the chain to the backing Buffer.
 */
function getBindingCount(value: unknown, name: string): number {
    if (isBuffer(value)) {
        return value.count;
    }
    if (isRawBuffer(value)) {
        // RawBuffer doesn't have a count; use byte length as fallback
        throw new Error(
            `Volten Error: Cannot infer thread count from RawBuffer "${name}" — ` +
                `RawBuffer doesn't track element count.\n` +
                `  Hint: Specify threads explicitly in the Kernel options.`
        );
    }
    if (isHandle(value)) {
        // Walk up the Handle chain to find the backing Buffer's count
        const source = (value as Handle)._node._bindings[
            (value as Handle)._name
        ];
        return getBindingCount(source, name);
    }
    throw new Error(
        `Volten Error: Cannot infer thread count from binding "${name}" — ` +
            `it is not a Buffer.`
    );
}

/**
 * Convert a 1D total thread count to workgroup dispatch dimensions.
 * Equivalent to threadsToDispatch3D([totalThreads, 1, 1], workgroupSize).
 *
 * Scalar threads are treated identically to [N, 1, 1] — all dispatch
 * paths use per-axis division for a single, consistent mental model.
 */
function threadsToDispatch(
    totalThreads: number,
    workgroupSize: [number, number, number]
): [number, number, number] {
    return threadsToDispatch3D([totalThreads, 1, 1], workgroupSize);
}

/**
 * Convert 3D total invocations to workgroup dispatch dimensions.
 * Each dimension is divided independently: dispatch[i] = ceil(threads[i] / workgroupSize[i])
 */
function threadsToDispatch3D(
    threads: [number, number, number],
    workgroupSize: [number, number, number]
): [number, number, number] {
    return [
        Math.ceil(threads[0] / workgroupSize[0]),
        Math.ceil(threads[1] / workgroupSize[1]),
        Math.ceil(threads[2] / workgroupSize[2])
    ];
}

/**
 * Normalize a 1D, 2D, or 3D thread count to 3D.
 * [x] -> [x, 1, 1]
 * [x, y] → [x, y, 1]
 * [x, y, z] → [x, y, z]
 */
function normalizeThreads(
    threads: [number] | [number, number] | [number, number, number]
): [number, number, number] {
    if (threads.length === 1) {
        return [threads[0], 1, 1];
    }
    if (threads.length === 2) {
        return [threads[0], threads[1], 1];
    }
    return threads;
}

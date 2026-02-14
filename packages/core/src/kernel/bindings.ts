// Binding generation for compute shaders
// Classifies user-provided bindings and generates WGSL declarations

import { Buffer } from '../data/buffer.js';
import { RawBuffer } from '../data/raw-buffer.js';
import { type Handle, isHandle } from '../graph/node.js';
import type { Kernel } from './kernel.js';

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
    /** WGSL access qualifier ("read" or "read_write") */
    readonly wgslAccess: string;
    /** The source object: Buffer, RawBuffer, or Handle */
    readonly source: Buffer | RawBuffer | Handle;
    /** Whether this is a Handle from a previous node */
    readonly isHandle: boolean;
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
 * Generate binding entries from user-provided bindings and kernel definition.
 * 
 * Classification rules:
 * - Buffer → var<storage, read|read_write> based on buffer.access
 * - RawBuffer → var<storage, read|read_write> based on buffer.access
 * - Handle → var<storage, read> (outputs from previous passes are read-only inputs)
 * - Anything else → throws an error
 * 
 * Validation:
 * - All kernel outputs must be present in bindings (v0: no pool allocator)
 * - No unknown types (numbers, strings, etc.)
 * 
 * @param bindings - User-provided bindings record
 * @param kernel - The kernel definition (for output validation)
 * @returns Array of classified binding entries
 */
export function generateBindings(
    bindings: Record<string, unknown>,
    kernel: Kernel
): BindingEntry[] {
    const entries: BindingEntry[] = [];
    let index = 0;

    // Validate that all kernel outputs are provided
    for (const output of kernel.outputs) {
        if (!(output.name in bindings)) {
            const providedKeys = Object.keys(bindings).join(', ');
            throw new Error(
                `Volten Error: Kernel declares output "${output.name}" but it was not provided in bindings.\n` +
                `  Provided bindings: { ${providedKeys} }\n` +
                `  Hint: In v0, all output buffers must be explicitly provided. ` +
                `Create a Buffer and pass it as "${output.name}" in your bindings.`
            );
        }
    }

    for (const [name, value] of Object.entries(bindings)) {
        if (isBuffer(value)) {
            entries.push({
                index: index++,
                name,
                wgslType: value.wgslType,
                wgslAccess: value.wgslAccess,
                source: value,
                isHandle: false,
            });
        } else if (isRawBuffer(value)) {
            entries.push({
                index: index++,
                name,
                wgslType: value.wgslType,
                wgslAccess: value.wgslAccess,
                source: value,
                isHandle: false,
            });
        } else if (isHandle(value)) {
            // Handles from previous passes are always read-only inputs
            // The actual type will be resolved from the source node's output buffer
            entries.push({
                index: index++,
                name,
                wgslType: 'array<f32>', // TODO: resolve actual type from Handle's source
                wgslAccess: 'read',
                source: value,
                isHandle: true,
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
                `  Expected: Buffer, RawBuffer, or Handle (output from a previous v.pass()).`
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
        .map(entry =>
            `@group(0) @binding(${entry.index}) var<storage, ${entry.wgslAccess}> ${entry.name}: ${entry.wgslType};`
        )
        .join('\n');
}

/**
 * Assemble the full shader source: binding declarations + kernel source.
 * 
 * The binding WGSL is prepended to the kernel's assembled source (which already
 * has builtin shorthands expanded and @compute/@workgroup_size injected).
 * 
 * @param kernel - The kernel definition
 * @param entries - Classified binding entries
 * @returns Complete WGSL shader source ready for pipeline creation
 */
export function assembleFullShader(
    kernel: Kernel,
    entries: BindingEntry[]
): string {
    const bindingWgsl = generateBindingWgsl(entries);
    const kernelSource = kernel.assembledSource;

    if (!bindingWgsl) {
        return kernelSource;
    }

    return `${bindingWgsl}\n\n${kernelSource}`;
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
export function resolveDispatch(
    kernel: Kernel,
    bindings: Record<string, unknown>,
    passThreads?: number | [number, number] | [number, number, number]
): [number, number, number] {
    const workgroupSize = kernel.workgroupSize;

    // 1. Pass-time override
    if (passThreads !== undefined) {
        if (typeof passThreads === 'number') {
            return threadsToDispatch(passThreads, workgroupSize);
        }
        // Array of total invocations per dimension → normalize to 3D and divide
        return threadsToDispatch3D(normalizeThreads(passThreads), workgroupSize);
    }

    // 2. Kernel-level threads spec
    const threads = kernel.threads;
    if (threads !== undefined) {
        if (typeof threads === 'number') {
            return threadsToDispatch(threads, workgroupSize);
        }
        if (typeof threads === 'string') {
            // Infer from named input
            const binding = bindings[threads];
            if (!binding) {
                throw new Error(
                    `Volten Error: Kernel threads spec references input "${threads}", ` +
                    `but it was not found in bindings.\n` +
                    `  Available bindings: ${Object.keys(bindings).join(', ')}`
                );
            }
            const count = getBindingCount(binding, threads);
            return threadsToDispatch(count, workgroupSize);
        }
        if (typeof threads === 'function') {
            const result = threads(bindings);
            // Function can return number, [n,n], or [n,n,n]
            if (typeof result === 'number') {
                return threadsToDispatch(result, workgroupSize);
            }
            return threadsToDispatch3D(normalizeThreads(result), workgroupSize);
        }
    }

    // 3. Auto-inference: find the first (and ideally only) buffer binding
    const bufferBindings = Object.entries(bindings).filter(
        ([, v]) => isBuffer(v) || isRawBuffer(v)
    );

    if (bufferBindings.length === 0) {
        throw new Error(
            'Volten Error: Cannot auto-infer thread count — no buffer bindings found.\n' +
            '  Hint: Specify threads explicitly: new Kernel(`...`, { threads: 1024 })'
        );
    }

    if (bufferBindings.length === 1) {
        const [name, buf] = bufferBindings[0];
        const count = getBindingCount(buf, name);
        return threadsToDispatch(count, workgroupSize);
    }

    // Multiple buffers — look for a kernel output to exclude, use remaining inputs
    const outputNames = new Set(kernel.outputNames);
    const inputBuffers = bufferBindings.filter(([name]) => !outputNames.has(name));

    if (inputBuffers.length === 1) {
        const [name, buf] = inputBuffers[0];
        const count = getBindingCount(buf, name);
        return threadsToDispatch(count, workgroupSize);
    }

    throw new Error(
        `Volten Error: Cannot auto-infer thread count — ${inputBuffers.length} input buffers found.\n` +
        `  Bindings: ${inputBuffers.map(([n]) => n).join(', ')}\n` +
        `  Hint: Specify threads explicitly: new Kernel(\`...\`, { threads: '${inputBuffers[0]?.[0] ?? 'input'}' })`
    );
}

/**
 * Get the element count from a binding value.
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
        Math.ceil(threads[2] / workgroupSize[2]),
    ];
}

/**
 * Normalize a 2D or 3D thread count to 3D.
 * [x, y] → [x, y, 1]
 * [x, y, z] → [x, y, z]
 */
function normalizeThreads(
    threads: [number, number] | [number, number, number]
): [number, number, number] {
    if (threads.length === 2) {
        return [threads[0], threads[1], 1];
    }
    return threads;
}

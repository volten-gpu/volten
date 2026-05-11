// Buffer class
// Manages structured data with automatic packing
// Lazy GPU buffer creation

import { type TypeDescriptor, getWgslType } from '../types/schema.js';
import { pack, getStride } from '../utils/alignment.js';
import { makeLabel } from '../utils/labels.js';
import { createResourceId } from './resource-identity.js';

/**
 * Buffer access mode (controls shader access, not CPU access)
 * - "r": Storage buffer that shaders can only read
 * - "rw": Storage buffer that shaders can read and write
 *
 * Note: CPU can always write to the buffer via device.queue.writeBuffer()
 * regardless of access mode — this is expected for updating constants.
 */
export type BufferAccess = 'r' | 'rw';

export interface BufferOptions {
    /** Optional human-friendly label for debugger/devtools usage. */
    label?: string;
}

/**
 * Buffer class for GPU storage buffers
 *
 * Handles automatic packing of JavaScript data to WGSL-compatible layout.
 * GPU buffer creation is lazy - call ensure(device) when ready to upload.
 *
 * @example
 * // Simple float array
 * const floats = new Buffer([1, 2, 3], "f32");
 *
 * // Vector array (note: vec3f has 16-byte stride due to alignment)
 * const positions = new Buffer([[0, 0, 0], [1, 1, 1]], "vec3f");
 *
 * // Struct array
 * const Particle = struct("Particle", {
 *   position: "vec3f",
 *   velocity: "vec3f",
 *   mass: "f32",
 * });
 * const particles = new Buffer([
 *   { position: [0, 0, 0], velocity: [1, 0, 0], mass: 1.0 },
 *   { position: [1, 0, 0], velocity: [0, 1, 0], mass: 2.0 },
 * ], Particle, "rw");
 */
export class Buffer {
    /** Stable identity for bind group caching. */
    readonly _resourceId = createResourceId();

    /** Human-friendly debug label */
    readonly label: string;

    /** The type descriptor for elements in this buffer */
    readonly type: TypeDescriptor;

    /** Access mode: "r" for read-only, "rw" for read-write */
    readonly access: BufferAccess;

    /** Number of elements in the buffer */
    readonly count: number;

    /** Bytes per element (includes alignment padding) */
    readonly stride: number;

    /** Total byte length of the buffer */
    readonly byteLength: number;

    /** Last CPU-provided packed data ready for GPU upload */
    private packedData: ArrayBuffer;

    /** Lazily created GPU buffer */
    private gpuBuffer: GPUBuffer | null = null;

    /** Bumped when the underlying GPUBuffer identity changes. */
    private gpuResourceVersion = 0;

    /** Device used to upload this buffer (captured on first ensure) */
    private device: GPUDevice | null = null;

    constructor(
        data: ArrayLike<unknown>,
        type: TypeDescriptor,
        access: BufferAccess = 'rw',
        options?: BufferOptions
    ) {
        this.label = makeLabel('Buffer', options?.label);
        this.type = type;
        this.access = access;
        this.count = data.length;
        this.stride = getStride(type);
        this.byteLength = this.stride * this.count;
        this.packedData = pack(data, type);
    }

    /**
     * Get the WGSL type string for this buffer's element type
     */
    get wgslType(): string {
        return `array<${getWgslType(this.type)}>`;
    }

    /**
     * Get the WGSL storage access qualifier
     */
    get wgslAccess(): string {
        return this.access === 'r' ? 'read' : 'read_write';
    }

    /**
     * Get last CPU-provided packed data (for testing/debugging).
     * GPU writes are not reflected here; use v.read() for GPU results.
     */
    get rawData(): ArrayBuffer {
        return this.packedData;
    }

    /**
     * Version of the underlying GPUBuffer allocation.
     * Content uploads do not change this; destroy/recreate does.
     */
    get _gpuResourceVersion(): number {
        return this.gpuResourceVersion;
    }

    /**
     * Ensure the GPU buffer exists, creating it if necessary
     *
     * @param device - GPUDevice to create the buffer on
     * @returns The GPU buffer
     */
    ensure(device: GPUDevice): GPUBuffer {
        this.device = device;

        if (this.gpuBuffer) {
            return this.gpuBuffer;
        }

        // Create GPU buffer with storage usage + copy flags for readback
        this.gpuBuffer = device.createBuffer({
            label: this.label,
            size: this.byteLength,
            usage:
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_SRC |
                GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });

        // Copy packed data to GPU
        const mapped = new Uint8Array(this.gpuBuffer.getMappedRange());
        mapped.set(new Uint8Array(this.packedData));
        this.gpuBuffer.unmap();

        return this.gpuBuffer;
    }

    /**
     * Replace the full CPU-provided contents without resizing the buffer.
     * If uploaded, the existing GPUBuffer is updated in place.
     */
    set(data: ArrayLike<unknown>): void {
        if (data.length !== this.count) {
            throw new Error(
                `Volten Error: Buffer "${this.label}" update changed element count ` +
                    `from ${this.count} to ${data.length}.\n` +
                    '  Buffer.set() can update contents, but it cannot resize the underlying GPU buffer.'
            );
        }

        this.packedData = pack(data, this.type);

        if (this.gpuBuffer && this.device) {
            this.device.queue.writeBuffer(this.gpuBuffer, 0, this.packedData);
        }
    }

    /**
     * Update a contiguous element range. Offset is measured in buffer elements.
     * If uploaded, the existing GPUBuffer is updated in place.
     */
    update(data: ArrayLike<unknown>, elementOffset = 0): void {
        this.validateUpdateRange(data.length, elementOffset);

        const packedUpdate = pack(data, this.type);
        const byteOffset = elementOffset * this.stride;
        new Uint8Array(this.packedData).set(
            new Uint8Array(packedUpdate),
            byteOffset
        );

        if (this.gpuBuffer && this.device) {
            this.device.queue.writeBuffer(
                this.gpuBuffer,
                byteOffset,
                packedUpdate
            );
        }
    }

    /**
     * Get the GPU buffer (throws if not yet created)
     */
    get gpuBufferOrThrow(): GPUBuffer {
        if (!this.gpuBuffer) {
            throw new Error(
                'Buffer has not been uploaded to GPU yet. Call ensure(device) first.'
            );
        }
        return this.gpuBuffer;
    }

    /**
     * Check if the GPU buffer has been created
     */
    get isUploaded(): boolean {
        return this.gpuBuffer !== null;
    }

    /**
     * Destroy the GPU buffer and release GPU memory
     */
    destroy(): void {
        if (this.gpuBuffer) {
            this.gpuBuffer.destroy();
            this.gpuBuffer = null;
            this.gpuResourceVersion++;
        }
    }

    private validateUpdateRange(length: number, elementOffset: number): void {
        if (!Number.isInteger(elementOffset) || elementOffset < 0) {
            throw new Error(
                `Volten Error: Buffer "${this.label}" update offset must be a non-negative integer.`
            );
        }
        if (elementOffset + length > this.count) {
            throw new Error(
                `Volten Error: Buffer "${this.label}" update range exceeds buffer length.\n` +
                    `  Buffer has ${this.count} elements, update starts at ${elementOffset} and contains ${length} elements.`
            );
        }
    }
}

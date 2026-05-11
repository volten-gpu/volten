// RawBuffer class
// Escape hatch for pre-packed data with literal WGSL type

import { type BufferAccess } from './buffer.js';
import { makeLabel } from '../utils/labels.js';
import { createResourceId } from './resource-identity.js';
import { asUint8Array, copyBytes, type ByteSource } from '../utils/bytes.js';

export interface RawBufferOptions {
    /** Optional human-friendly label for debugger/devtools usage. */
    label?: string;
}

/**
 * RawBuffer for pre-packed data
 *
 * Use this when you've already packed your data or need a custom type
 * string that Volten's type system doesn't support.
 *
 * @example
 * // Pre-packed data with custom type
 * const raw = new RawBuffer(myPackedArrayBuffer, "array<MyCustomStruct, 100>");
 *
 * // Trailing array struct (special WGSL pattern)
 * const buf = new RawBuffer(packedData, "ParticleBuffer");
 */
export class RawBuffer {
    /** Stable identity for bind group caching. */
    readonly _resourceId = createResourceId();

    /** Human-friendly debug label */
    readonly label: string;

    /** Literal WGSL type string */
    readonly wgslType: string;

    /** Access mode: "r" for read-only, "rw" for read-write */
    readonly access: BufferAccess;

    /** Total byte length */
    readonly byteLength: number;

    /** Last CPU-provided pre-packed data */
    private packedData: ArrayBuffer;

    /** Lazily created GPU buffer */
    private gpuBuffer: GPUBuffer | null = null;

    /** Bumped when the underlying GPUBuffer identity changes. */
    private gpuResourceVersion = 0;

    /** Device used to upload this buffer (captured on first ensure) */
    private device: GPUDevice | null = null;

    constructor(
        data: ArrayBuffer,
        wgslType: string,
        access: BufferAccess = 'rw',
        options?: RawBufferOptions
    ) {
        this.label = makeLabel('RawBuffer', options?.label);
        this.packedData = data;
        this.wgslType = wgslType;
        this.access = access;
        this.byteLength = data.byteLength;
    }

    /**
     * Get the WGSL storage access qualifier
     */
    get wgslAccess(): string {
        return this.access === 'r' ? 'read' : 'read_write';
    }

    /**
     * Get last CPU-provided packed data.
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
     */
    ensure(device: GPUDevice): GPUBuffer {
        this.device = device;

        if (this.gpuBuffer) {
            return this.gpuBuffer;
        }

        this.gpuBuffer = device.createBuffer({
            label: this.label,
            size: this.byteLength,
            usage:
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_SRC |
                GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });

        const mapped = new Uint8Array(this.gpuBuffer.getMappedRange());
        mapped.set(new Uint8Array(this.packedData));
        this.gpuBuffer.unmap();

        return this.gpuBuffer;
    }

    /**
     * Replace the full CPU-provided contents without resizing the buffer.
     * If uploaded, the existing GPUBuffer is updated in place.
     */
    set(data: ByteSource): void {
        const bytes = asUint8Array(data);
        if (bytes.byteLength !== this.byteLength) {
            throw new Error(
                `Volten Error: RawBuffer "${this.label}" update changed byte length ` +
                    `from ${this.byteLength} to ${bytes.byteLength} bytes.\n` +
                    '  RawBuffer.set() can update contents, but it cannot resize the underlying GPU buffer.'
            );
        }

        this.packedData = copyBytes(bytes);

        if (this.gpuBuffer && this.device) {
            this.device.queue.writeBuffer(this.gpuBuffer, 0, this.packedData);
        }
    }

    /**
     * Update a byte range. Offset is measured in bytes.
     * If uploaded, the existing GPUBuffer is updated in place.
     */
    update(data: ByteSource, byteOffset = 0): void {
        const bytes = asUint8Array(data);
        this.validateUpdateRange(bytes.byteLength, byteOffset);

        new Uint8Array(this.packedData).set(bytes, byteOffset);

        if (this.gpuBuffer && this.device) {
            this.device.queue.writeBuffer(
                this.gpuBuffer,
                byteOffset,
                copyBytes(bytes)
            );
        }
    }

    /**
     * Get the GPU buffer (throws if not yet created)
     */
    get gpuBufferOrThrow(): GPUBuffer {
        if (!this.gpuBuffer) {
            throw new Error(
                'RawBuffer has not been uploaded to GPU yet. Call ensure(device) first.'
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

    private validateUpdateRange(length: number, byteOffset: number): void {
        if (!Number.isInteger(byteOffset) || byteOffset < 0) {
            throw new Error(
                `Volten Error: RawBuffer "${this.label}" update offset must be a non-negative integer.`
            );
        }
        if (byteOffset + length > this.byteLength) {
            throw new Error(
                `Volten Error: RawBuffer "${this.label}" update range exceeds buffer length.\n` +
                    `  RawBuffer has ${this.byteLength} bytes, update starts at ${byteOffset} and contains ${length} bytes.`
            );
        }
    }
}

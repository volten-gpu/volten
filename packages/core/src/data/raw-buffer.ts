// RawBuffer class
// Escape hatch for pre-packed data with literal WGSL type

import { type BufferAccess } from './buffer.js';
import { makeLabel } from '../utils/labels.js';

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
    /** Human-friendly debug label */
    readonly label: string;

    /** Literal WGSL type string */
    readonly wgslType: string;

    /** Access mode: "r" for read-only, "rw" for read-write */
    readonly access: BufferAccess;

    /** Total byte length */
    readonly byteLength: number;

    /** Pre-packed data */
    private readonly packedData: ArrayBuffer;

    /** Lazily created GPU buffer */
    private gpuBuffer: GPUBuffer | null = null;

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
     * Get raw packed data
     */
    get rawData(): ArrayBuffer {
        return this.packedData;
    }

    /**
     * Ensure the GPU buffer exists, creating it if necessary
     */
    ensure(device: GPUDevice): GPUBuffer {
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
            mappedAtCreation: true,
        });

        const mapped = new Uint8Array(this.gpuBuffer.getMappedRange());
        mapped.set(new Uint8Array(this.packedData));
        this.gpuBuffer.unmap();

        return this.gpuBuffer;
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
        }
    }
}

// Buffer class
// Manages structured data with automatic packing
// Lazy GPU buffer creation

import { type TypeDescriptor, resolveType, getWgslType } from '../types/schema.js';
import { pack, getStride } from '../utils/alignment.js';

/**
 * Buffer access mode (controls shader access, not CPU access)
 * - "r": Storage buffer that shaders can only read
 * - "rw": Storage buffer that shaders can read and write
 * 
 * Note: CPU can always write to the buffer via device.queue.writeBuffer()
 * regardless of access mode — this is expected for updating constants.
 */
export type BufferAccess = 'r' | 'rw';

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

    /** Packed data ready for GPU upload */
    private readonly packedData: ArrayBuffer;

    /** Lazily created GPU buffer */
    private gpuBuffer: GPUBuffer | null = null;

    constructor(
        data: ArrayLike<unknown>,
        type: TypeDescriptor,
        access: BufferAccess = 'rw'
    ) {
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
     * Get raw packed data (for testing/debugging)
     */
    get rawData(): ArrayBuffer {
        return this.packedData;
    }

    /**
     * Ensure the GPU buffer exists, creating it if necessary
     * 
     * @param device - GPUDevice to create the buffer on
     * @returns The GPU buffer
     */
    ensure(device: GPUDevice): GPUBuffer {
        if (this.gpuBuffer) {
            return this.gpuBuffer;
        }

        // Create GPU buffer with storage usage + copy flags for readback
        this.gpuBuffer = device.createBuffer({
            size: this.byteLength,
            usage:
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_SRC |
                GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });

        // Copy packed data to GPU
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
        }
    }
}

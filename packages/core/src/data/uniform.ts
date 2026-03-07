// Uniform class
// Manages single-value uniform data with automatic packing

import { type TypeDescriptor, getWgslType } from '../types/schema.js';
import { pack } from '../utils/alignment.js';
import {
    getUniformPackingAddressSpace,
    type UniformLayoutMode,
} from '../utils/uniform-layout.js';

/**
 * Uniform class for GPU uniform buffers.
 *
 * Handles automatic packing of JavaScript data to WGSL-compatible layout.
 * GPU buffer creation is lazy - call ensure(device) when ready to upload.
 *
 * @example
 * const multiplier = new Uniform(2.0, 'f32');
 * const transform = new Uniform([
 *   1, 0, 0, 0,
 *   0, 1, 0, 0,
 *   0, 0, 1, 0,
 *   0, 0, 0, 1,
 * ], 'mat4x4f');
 */
export class Uniform {
    /** The type descriptor for this uniform value */
    readonly type: TypeDescriptor;

    /** Last JavaScript value provided by user */
    private value: unknown;

    /** Active uniform layout mode for packing */
    private layoutMode: UniformLayoutMode = 'classic';

    /** Packed data ready for GPU upload */
    private packedData: ArrayBuffer;

    /** Lazily created GPU buffer */
    private gpuBuffer: GPUBuffer | null = null;

    /** Device used to upload this uniform (captured on first ensure) */
    private device: GPUDevice | null = null;

    constructor(
        data: unknown,
        type: TypeDescriptor
    ) {
        this.type = type;
        this.value = data;
        this.packedData = this.packOne(data);
    }

    /**
     * Bytes for one packed uniform value (includes alignment padding).
     */
    get byteLength(): number {
        return this.packedData.byteLength;
    }

    /**
     * Get the WGSL type string for this uniform value
     */
    get wgslType(): string {
        return getWgslType(this.type);
    }

    /**
     * Get raw packed data (for testing/debugging)
     */
    get rawData(): ArrayBuffer {
        return this.packedData;
    }

    /**
     * Current resolved uniform packing mode for this instance.
     */
    get resolvedLayoutMode(): UniformLayoutMode {
        return this.layoutMode;
    }

    /**
     * Set packing mode. Used by VoltenContext to keep CPU packing in sync
     * with generated WGSL layout strategy.
     */
    setLayoutMode(mode: UniformLayoutMode): void {
        if (mode === this.layoutMode) {
            return;
        }
        this.layoutMode = mode;
        this.repackAndUploadIfNeeded();
    }

    /**
     * Ensure the GPU buffer exists, creating it if necessary.
     *
     * @param device - GPUDevice to create the buffer on
     * @returns The GPU buffer
     */
    ensure(device: GPUDevice): GPUBuffer {
        this.device = device;

        if (this.gpuBuffer) {
            return this.gpuBuffer;
        }

        this.gpuBuffer = this.createAndUpload(device);
        return this.gpuBuffer;
    }

    /**
     * Update the uniform value.
     *
     * If the uniform is already uploaded, this writes the new bytes to the GPU
     * immediately via queue.writeBuffer(). If not uploaded yet, it updates only
     * the local packed bytes and the first ensure() will upload them.
     */
    set(data: unknown): void {
        this.value = data;
        this.repackAndUploadIfNeeded();
    }

    /**
     * Get the GPU buffer (throws if not yet created)
     */
    get gpuBufferOrThrow(): GPUBuffer {
        if (!this.gpuBuffer) {
            throw new Error(
                'Uniform has not been uploaded to GPU yet. Call ensure(device) first.'
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

    /**
     * Pack exactly one value and validate resulting byte size.
     */
    private packOne(data: unknown): ArrayBuffer {
        return pack([data], this.type, {
            addressSpace: getUniformPackingAddressSpace(this.layoutMode),
        });
    }

    private repackAndUploadIfNeeded(): void {
        const previousByteLength = this.byteLength;
        this.packedData = this.packOne(this.value);

        if (!this.gpuBuffer || !this.device) {
            return;
        }

        if (this.byteLength !== previousByteLength) {
            this.gpuBuffer.destroy();
            this.gpuBuffer = this.createAndUpload(this.device);
            return;
        }

        this.device.queue.writeBuffer(this.gpuBuffer, 0, this.packedData);
    }

    private createAndUpload(device: GPUDevice): GPUBuffer {
        const buffer = device.createBuffer({
            size: this.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });

        const mapped = new Uint8Array(buffer.getMappedRange());
        mapped.set(new Uint8Array(this.packedData));
        buffer.unmap();
        return buffer;
    }
}

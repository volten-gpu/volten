// Uniform class
// Manages single-value uniform data with automatic packing

import { type TypeDescriptor, getWgslType } from '../types/schema.js';
import { pack } from '../utils/alignment.js';
import {
    getUniformPackingLayoutRules,
    type UniformLayoutMode
} from '../utils/uniform-layout.js';
import { makeLabel } from '../utils/labels.js';
import { createResourceId } from './resource-identity.js';

export interface UniformOptions {
    /** Optional human-friendly label for debugger/devtools usage. */
    label?: string;
}

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
    /** Stable identity for bind group caching. */
    readonly _resourceId = createResourceId();

    /** Human-friendly debug label */
    readonly label: string;

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

    /** Bumped when the underlying GPUBuffer identity changes. */
    private gpuResourceVersion = 0;

    /** Device used to upload this uniform (captured on first ensure) */
    private device: GPUDevice | null = null;

    constructor(data: unknown, type: TypeDescriptor, options?: UniformOptions) {
        this.label = makeLabel('Uniform', options?.label);
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
     * Version of the underlying GPUBuffer allocation.
     * Content uploads do not change this; destroy/recreate does.
     */
    get _gpuResourceVersion(): number {
        return this.gpuResourceVersion;
    }

    /**
     * Current resolved uniform packing mode for this instance.
     */
    get resolvedLayoutMode(): UniformLayoutMode {
        return this.layoutMode;
    }

    /**
     * Bind packing mode. Used by VoltenContext to keep CPU packing in sync
     * with generated WGSL layout strategy.
     */
    setLayoutMode(mode: UniformLayoutMode): void {
        if (mode === this.layoutMode) {
            return;
        }
        if (this.gpuBuffer) {
            throw new Error(
                `Volten Error: Uniform "${this.label}" was already uploaded with ${this.layoutMode} layout, ` +
                    `but this context requires ${mode} layout.\n` +
                    '  Create a separate Uniform for contexts with different uniform layout modes.'
            );
        }
        this.layoutMode = mode;
        this.packedData = this.packOne(this.value);
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
        const packedData = this.packOne(data);
        if (packedData.byteLength !== this.byteLength) {
            throw new Error(
                `Volten Error: Uniform "${this.label}" update changed packed byte length ` +
                    `from ${this.byteLength} to ${packedData.byteLength} bytes.\n` +
                    '  Uniform.set() can update contents, but it cannot resize the underlying GPU buffer.'
            );
        }

        this.value = data;
        this.packedData = packedData;

        if (this.gpuBuffer && this.device) {
            this.device.queue.writeBuffer(this.gpuBuffer, 0, this.packedData);
        }
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
            this.gpuResourceVersion++;
        }
    }

    /**
     * Pack exactly one value and validate resulting byte size.
     */
    private packOne(data: unknown): ArrayBuffer {
        return pack([data], this.type, {
            layoutRules: getUniformPackingLayoutRules(this.layoutMode)
        });
    }

    private createAndUpload(device: GPUDevice): GPUBuffer {
        const buffer = device.createBuffer({
            label: this.label,
            size: this.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });

        const mapped = new Uint8Array(buffer.getMappedRange());
        mapped.set(new Uint8Array(this.packedData));
        buffer.unmap();
        return buffer;
    }
}

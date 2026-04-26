import { RawBuffer } from '../data/raw-buffer.js';
import type { ResolvedDebugOptions } from './types.js';

export const VOLTEN_DEBUG_BUFFER_NAME =
    '_volten_debug_buffer';
export const VOLTEN_DEBUG_BUFFER_STRUCT_NAME =
    '_volten_debug_storage_buffer';

const DEBUG_RESET_HEADER = new Uint32Array([0, 0]);

export class DebugBufferResource {
    readonly buffer: RawBuffer;
    readonly bufferSize: number;
    readonly capacityWords: number;

    constructor(options: ResolvedDebugOptions, label: string) {
        this.bufferSize = options.bufferSize;
        this.capacityWords = options.capacityWords;
        this.buffer = new RawBuffer(
            new ArrayBuffer(DEBUG_RESET_HEADER.byteLength + options.bufferSize),
            VOLTEN_DEBUG_BUFFER_STRUCT_NAME,
            'rw',
            { label }
        );
    }

    reset(device: GPUDevice): void {
        const gpuBuffer = this.buffer.ensure(device);
        device.queue.writeBuffer(gpuBuffer, 0, DEBUG_RESET_HEADER);
    }

    destroy(): void {
        this.buffer.destroy();
    }
}

export interface NodeDebugState {
    readonly resource: DebugBufferResource;
    readonly messages: readonly string[];
}

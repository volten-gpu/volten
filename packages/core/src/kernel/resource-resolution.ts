import { Buffer } from '../data/buffer.js';
import { RawBuffer } from '../data/raw-buffer.js';
import { Uniform } from '../data/uniform.js';
import { type Handle, isHandle } from '../graph/node.js';

export type BindableResource = Buffer | RawBuffer | Uniform;

/**
 * Walk a Handle chain to find the root Buffer or RawBuffer.
 *
 * Returns null for non-buffer bindings so compiler/scheduler code can ignore
 * uniforms and future non-buffer resources.
 */
export function resolveConcreteBuffer(
    value: unknown
): Buffer | RawBuffer | null {
    if (value instanceof Buffer || value instanceof RawBuffer) {
        return value;
    }
    if (isHandle(value)) {
        const parentBinding = value._node._bindings[value._name];
        return resolveConcreteBuffer(parentBinding);
    }
    return null;
}

export function resolveBindableResource(
    value: Buffer | RawBuffer | Uniform | Handle
): BindableResource {
    if (
        value instanceof Buffer ||
        value instanceof RawBuffer ||
        value instanceof Uniform
    ) {
        return value;
    }

    const sourceBinding = value._node._bindings[value._name];
    return resolveBindableResource(
        sourceBinding as Buffer | RawBuffer | Uniform | Handle
    );
}

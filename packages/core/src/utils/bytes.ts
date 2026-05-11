export type ByteSource = ArrayBuffer | ArrayBufferView;

export function asUint8Array(data: ByteSource): Uint8Array {
    if (data instanceof ArrayBuffer) {
        return new Uint8Array(data);
    }
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

export function copyBytes(data: ByteSource): ArrayBuffer {
    const bytes = asUint8Array(data);
    const copy = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(copy).set(bytes);
    return copy;
}

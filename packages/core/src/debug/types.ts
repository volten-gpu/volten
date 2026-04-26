export const DEBUG_HEADER_WORDS = 2;
export const DEBUG_RECORD_HEADER_WORDS = 6;
export const DEFAULT_DEBUG_BUFFER_SIZE = 16 * 1024;

export const DEBUG_KIND_TAGS = {
    f32: 1,
    u32: 2,
    i32: 3,
    vec2f: 4,
    vec3f: 5,
    vec4f: 6,
    mat4x4f: 7
} as const;

export const DEBUG_KIND_WORD_COUNTS = {
    f32: 1,
    u32: 1,
    i32: 1,
    vec2f: 2,
    vec3f: 3,
    vec4f: 4,
    mat4x4f: 16
} as const;

export type DebugValueKind = keyof typeof DEBUG_KIND_TAGS;

export interface DebugOptions {
    /**
     * Total payload size available to debug records, in bytes.
     * This excludes the hidden atomic header used internally by Volten.
     */
    bufferSize?: number;
}

export interface ResolvedDebugOptions {
    readonly bufferSize: number;
    readonly capacityWords: number;
}

export interface DebugLog {
    readonly kind: DebugValueKind;
    readonly gid: readonly [number, number, number];
    readonly message?: string;
    readonly value: number | number[];
}

export interface DebugReadResult {
    readonly logs: readonly DebugLog[];
    readonly dropped: number;
    readonly usedWords: number;
    readonly truncated: boolean;
    readonly bufferSize: number;
}

export function resolveDebugOptions(
    debug?: boolean | DebugOptions
): ResolvedDebugOptions | null {
    if (!debug) {
        return null;
    }

    const requestedBufferSize =
        typeof debug === 'object' ? debug.bufferSize : undefined;
    const bufferSize = requestedBufferSize ?? DEFAULT_DEBUG_BUFFER_SIZE;

    if (!Number.isFinite(bufferSize) || bufferSize <= 0) {
        throw new Error(
            'Volten Error: debug.bufferSize must be a positive finite number of bytes.'
        );
    }

    const normalizedBytes = Math.ceil(bufferSize / 4) * 4;

    return {
        bufferSize: normalizedBytes,
        capacityWords: normalizedBytes / 4
    };
}

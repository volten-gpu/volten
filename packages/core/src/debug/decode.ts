import {
    DEBUG_HEADER_WORDS,
    DEBUG_KIND_TAGS,
    type DebugLog,
    type DebugReadResult,
    type DebugValueKind
} from './types.js';

const TAG_TO_KIND = new Map<number, DebugValueKind>(
    Object.entries(DEBUG_KIND_TAGS).map(([kind, tag]) => [
        tag,
        kind as DebugValueKind
    ])
);

function decodeWordAsF32(word: number): number {
    const view = new DataView(new ArrayBuffer(4));
    view.setUint32(0, word, true);
    return view.getFloat32(0, true);
}

function decodeWordAsI32(word: number): number {
    const view = new DataView(new ArrayBuffer(4));
    view.setUint32(0, word, true);
    return view.getInt32(0, true);
}

function decodePayload(
    kind: DebugValueKind,
    payload: Uint32Array
): number | number[] {
    switch (kind) {
        case 'f32':
            return decodeWordAsF32(payload[0]);
        case 'u32':
            return payload[0] ?? 0;
        case 'i32':
            return decodeWordAsI32(payload[0]);
        case 'vec2f':
        case 'vec3f':
        case 'vec4f':
        case 'mat4x4f':
            return Array.from(payload, (word) => decodeWordAsF32(word));
    }
}

function formatDebugScalar(value: number): string {
    return Object.is(value, -0) ? '-0' : String(value);
}

function formatDebugValue(value: number | number[]): string {
    if (Array.isArray(value)) {
        return value.map(formatDebugScalar).join(', ');
    }

    return formatDebugScalar(value);
}

function formatDebugGid(gid: readonly [number, number, number]): string {
    return `[${gid.join(',')}]`;
}

function printDebugLogs(logs: readonly DebugLog[]): void {
    for (const log of logs) {
        const message = log.message ? `${log.message}: ` : '';
        console.log(
            `${formatDebugGid(log.gid)} ${message}${formatDebugValue(log.value)}`
        );
    }
}

export function decodeDebugBuffer(
    raw: ArrayBuffer,
    messages: readonly string[],
    bufferSize: number
): DebugReadResult {
    const words = new Uint32Array(raw);
    const attemptedUsedWords = words[0] ?? 0;
    const dropped = words[1] ?? 0;
    const availableWords = Math.max(0, words.length - DEBUG_HEADER_WORDS);
    const usedWords = Math.min(attemptedUsedWords, availableWords);
    const truncated = attemptedUsedWords > availableWords;

    const logs: DebugLog[] = [];
    let cursor = DEBUG_HEADER_WORDS;
    const payloadLimit = DEBUG_HEADER_WORDS + usedWords;

    while (cursor + 5 < payloadLimit) {
        const tag = words[cursor + 0];
        const messageId = words[cursor + 1];
        const payloadWords = words[cursor + 5];
        const recordEnd = cursor + 6 + payloadWords;

        if (recordEnd > payloadLimit) {
            break;
        }

        const kind = TAG_TO_KIND.get(tag);
        if (!kind) {
            break;
        }

        const payload = words.slice(cursor + 6, recordEnd);
        logs.push({
            kind,
            gid: [
                words[cursor + 2] ?? 0,
                words[cursor + 3] ?? 0,
                words[cursor + 4] ?? 0
            ],
            message:
                messageId === 0
                    ? undefined
                    : (messages[messageId - 1] ?? undefined),
            value: decodePayload(kind, payload)
        });

        cursor = recordEnd;
    }

    return {
        logs,
        dropped,
        usedWords,
        truncated,
        bufferSize,
        print: () => printDebugLogs(logs)
    };
}

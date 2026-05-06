import type { ShaderTransform } from '../kernel/shader.js';
import {
    isIdentifierPart,
    isIdentifierStart,
    parseParenthesizedSpan,
    skipWhitespace,
    splitTopLevelCommaList
} from '../kernel/wgsl-source.js';
import {
    DEBUG_KIND_TAGS,
    DEBUG_KIND_WORD_COUNTS,
    DEBUG_RECORD_HEADER_WORDS,
    type DebugValueKind
} from './types.js';
import { VOLTEN_DEBUG_BUFFER_NAME } from './resource.js';

const VOLTEN_DEBUG_ENABLED_NAME = '_volten_debug_enabled';
const VOLTEN_DEBUG_GID_NAME = '_volten_debug_gid';
const VOLTEN_DEBUG_BEGIN_FN = '_volten_debug_begin_invocation';
const VOLTEN_DEBUG_RESERVE_FN = '_volten_debug_reserve';
const VOLTEN_DEBUG_WRITE_HEADER_FN = '_volten_debug_write_header';
const VOLTEN_DEBUG_OVERFLOW_SENTINEL = '0xffffffffu';

type SupportedDebugFunctionName =
    | 'debugF32'
    | 'debugU32'
    | 'debugI32'
    | 'debugVec2'
    | 'debugVec2f'
    | 'debugVec3'
    | 'debugVec3f'
    | 'debugVec4'
    | 'debugVec4f'
    | 'debugMat4'
    | 'debugMat4x4f';

const DEBUG_FUNCTIONS: Record<SupportedDebugFunctionName, DebugValueKind> = {
    debugF32: 'f32',
    debugU32: 'u32',
    debugI32: 'i32',
    debugVec2: 'vec2f',
    debugVec2f: 'vec2f',
    debugVec3: 'vec3f',
    debugVec3f: 'vec3f',
    debugVec4: 'vec4f',
    debugVec4f: 'vec4f',
    debugMat4: 'mat4x4f',
    debugMat4x4f: 'mat4x4f'
};

const DEBUG_FUNCTION_NAMES = new Set<string>(Object.keys(DEBUG_FUNCTIONS));

interface ScanResult {
    readonly replacement: string;
    readonly endIndex: number;
}

interface ParsedCall {
    readonly endIndex: number;
}

export interface DebugShaderTransform extends ShaderTransform {
    readonly messages: readonly string[];
}

// for debug messages like: debugF32("message", value)
// this function will be used to evaluate the "message" part,
// It validates that the message is a double-quoted literal,
// decodes simple escapes like \n, \", \\, and returns the actual
// string.
function decodeStringLiteral(literal: string): string {
    const trimmed = literal.trim();
    if (
        trimmed.length < 2 ||
        trimmed[0] !== '"' ||
        trimmed[trimmed.length - 1] !== '"'
    ) {
        throw new Error(
            `Volten Error: Debug messages must use double-quoted string literals. Received: ${trimmed}`
        );
    }

    let result = '';
    for (let i = 1; i < trimmed.length - 1; i++) {
        const char = trimmed[i];
        if (char !== '\\') {
            result += char;
            continue;
        }

        const next = trimmed[++i];
        switch (next) {
            case '"':
                result += '"';
                break;
            case '\\':
                result += '\\';
                break;
            case 'n':
                result += '\n';
                break;
            case 'r':
                result += '\r';
                break;
            case 't':
                result += '\t';
                break;
            case '0':
                result += '\0';
                break;
            default:
                throw new Error(
                    `Volten Error: Unsupported escape sequence "\\${next}" in debug message.`
                );
        }
    }
    return result;
}

function createMessageRegistry() {
    const messages: string[] = [];
    const ids = new Map<string, number>();

    return {
        messages,
        getId(message: string): number {
            const existing = ids.get(message);
            if (existing !== undefined) {
                return existing;
            }

            const nextId = messages.length + 1;
            ids.set(message, nextId);
            messages.push(message);
            return nextId;
        }
    };
}

function validateEnableDebugCall(
    source: string,
    callName: string,
    openParenIndex: number
): ParsedCall {
    const { content: args, endIndex } = parseParenthesizedSpan(
        source,
        openParenIndex,
        {
            unterminatedMessage:
                'Volten Error: Unterminated debug call while parsing shader source.'
        }
    );
    if (args.trim().length > 0) {
        throw new Error(
            `Volten Error: ${callName}() does not accept any arguments.`
        );
    }

    return {
        endIndex
    };
}

function rewriteTypedDebugCall(
    source: string,
    callName: string,
    openParenIndex: number,
    messages: ReturnType<typeof createMessageRegistry>
): ScanResult {
    const { content: args, endIndex } = parseParenthesizedSpan(
        source,
        openParenIndex,
        {
            unterminatedMessage:
                'Volten Error: Unterminated debug call while parsing shader source.'
        }
    );
    const parsedArgs = splitTopLevelCommaList(args);

    let messageId = 0;
    let valueExpression: string | undefined;

    if (parsedArgs.length === 1) {
        valueExpression = parsedArgs[0];
    } else if (parsedArgs.length === 2) {
        messageId = messages.getId(decodeStringLiteral(parsedArgs[0]));
        valueExpression = parsedArgs[1];
    } else {
        throw new Error(
            `Volten Error: ${callName}() expects either (value) or ("message", value).`
        );
    }

    if (!valueExpression || valueExpression.trim().length === 0) {
        throw new Error(`Volten Error: ${callName}() requires a value to log.`);
    }

    return {
        replacement: `${callName}(${messageId}u, ${valueExpression.trim()})`,
        endIndex
    };
}

function rewriteDebugSource(source: string): {
    source: string;
    messages: readonly string[];
} {
    const registry = createMessageRegistry();
    let index = 0;
    let output = '';

    while (index < source.length) {
        const char = source[index];
        const next = source[index + 1];

        if (char === '/' && next === '/') {
            const end = source.indexOf('\n', index);
            const sliceEnd = end === -1 ? source.length : end;
            output += source.slice(index, sliceEnd);
            index = sliceEnd;
            continue;
        }

        if (char === '/' && next === '*') {
            const end = source.indexOf('*/', index + 2);
            const sliceEnd = end === -1 ? source.length : end + 2;
            output += source.slice(index, sliceEnd);
            index = sliceEnd;
            continue;
        }

        // looks for identifier starts, like variable names,
        // or "fn"
        if (!isIdentifierStart(char)) {
            output += char;
            index++;
            continue;
        }

        // and tries to find the start and end point of
        // the identifier. Ideally we're looking for
        // "debugF32" or similar
        let end = index + 1;
        while (end < source.length && isIdentifierPart(source[end])) {
            end++;
        }

        const identifier = source.slice(index, end);
        const afterIdentifier = skipWhitespace(source, end);
        const isCall = source[afterIdentifier] === '(';

        if (!isCall) {
            output += identifier;
            index = end;
            continue;
        }

        if (identifier === 'debug') {
            throw new Error(
                'Volten Error: Generic debug(...) is not supported in v0. Use typed helpers like debugF32(), debugVec3(), or debugMat4().'
            );
        }

        if (identifier === 'enableDebug') {
            const parsed = validateEnableDebugCall(
                source,
                identifier,
                afterIdentifier
            );
            output += source.slice(index, parsed.endIndex);
            index = parsed.endIndex;
            continue;
        }

        if (DEBUG_FUNCTION_NAMES.has(identifier)) {
            const rewritten = rewriteTypedDebugCall(
                source,
                identifier,
                afterIdentifier,
                registry
            );
            output += rewritten.replacement;
            index = rewritten.endIndex;
            continue;
        }

        output += identifier;
        index = end;
    }

    return {
        source: output,
        messages: registry.messages
    };
}

function createDebugFunctionHelperWgsl(
    name: SupportedDebugFunctionName,
    kind: DebugValueKind
): string {
    const kindTag = DEBUG_KIND_TAGS[kind];
    const payloadWords = DEBUG_KIND_WORD_COUNTS[kind];

    switch (kind) {
        case 'f32':
            return `fn ${name}(messageId: u32, value: f32) {
    if (!${VOLTEN_DEBUG_ENABLED_NAME}) {
        return;
    }

    let start = ${VOLTEN_DEBUG_RESERVE_FN}(${payloadWords}u);
    if (start == ${VOLTEN_DEBUG_OVERFLOW_SENTINEL}) {
        return;
    }

    ${VOLTEN_DEBUG_WRITE_HEADER_FN}(start, ${kindTag}u, messageId, ${payloadWords}u);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u] = bitcast<u32>(value);
}`;
        case 'u32':
            return `fn ${name}(messageId: u32, value: u32) {
    if (!${VOLTEN_DEBUG_ENABLED_NAME}) {
        return;
    }

    let start = ${VOLTEN_DEBUG_RESERVE_FN}(${payloadWords}u);
    if (start == ${VOLTEN_DEBUG_OVERFLOW_SENTINEL}) {
        return;
    }

    ${VOLTEN_DEBUG_WRITE_HEADER_FN}(start, ${kindTag}u, messageId, ${payloadWords}u);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u] = value;
}`;
        case 'i32':
            return `fn ${name}(messageId: u32, value: i32) {
    if (!${VOLTEN_DEBUG_ENABLED_NAME}) {
        return;
    }

    let start = ${VOLTEN_DEBUG_RESERVE_FN}(${payloadWords}u);
    if (start == ${VOLTEN_DEBUG_OVERFLOW_SENTINEL}) {
        return;
    }

    ${VOLTEN_DEBUG_WRITE_HEADER_FN}(start, ${kindTag}u, messageId, ${payloadWords}u);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u] = bitcast<u32>(value);
}`;
        case 'vec2f':
            return `fn ${name}(messageId: u32, value: vec2<f32>) {
    if (!${VOLTEN_DEBUG_ENABLED_NAME}) {
        return;
    }

    let start = ${VOLTEN_DEBUG_RESERVE_FN}(${payloadWords}u);
    if (start == ${VOLTEN_DEBUG_OVERFLOW_SENTINEL}) {
        return;
    }

    ${VOLTEN_DEBUG_WRITE_HEADER_FN}(start, ${kindTag}u, messageId, ${payloadWords}u);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 0u] = bitcast<u32>(value.x);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 1u] = bitcast<u32>(value.y);
}`;
        case 'vec3f':
            return `fn ${name}(messageId: u32, value: vec3<f32>) {
    if (!${VOLTEN_DEBUG_ENABLED_NAME}) {
        return;
    }

    let start = ${VOLTEN_DEBUG_RESERVE_FN}(${payloadWords}u);
    if (start == ${VOLTEN_DEBUG_OVERFLOW_SENTINEL}) {
        return;
    }

    ${VOLTEN_DEBUG_WRITE_HEADER_FN}(start, ${kindTag}u, messageId, ${payloadWords}u);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 0u] = bitcast<u32>(value.x);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 1u] = bitcast<u32>(value.y);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 2u] = bitcast<u32>(value.z);
}`;
        case 'vec4f':
            return `fn ${name}(messageId: u32, value: vec4<f32>) {
    if (!${VOLTEN_DEBUG_ENABLED_NAME}) {
        return;
    }

    let start = ${VOLTEN_DEBUG_RESERVE_FN}(${payloadWords}u);
    if (start == ${VOLTEN_DEBUG_OVERFLOW_SENTINEL}) {
        return;
    }

    ${VOLTEN_DEBUG_WRITE_HEADER_FN}(start, ${kindTag}u, messageId, ${payloadWords}u);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 0u] = bitcast<u32>(value.x);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 1u] = bitcast<u32>(value.y);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 2u] = bitcast<u32>(value.z);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 3u] = bitcast<u32>(value.w);
}`;
        case 'mat4x4f':
            return `fn ${name}(messageId: u32, value: mat4x4<f32>) {
    if (!${VOLTEN_DEBUG_ENABLED_NAME}) {
        return;
    }

    let start = ${VOLTEN_DEBUG_RESERVE_FN}(${payloadWords}u);
    if (start == ${VOLTEN_DEBUG_OVERFLOW_SENTINEL}) {
        return;
    }

    ${VOLTEN_DEBUG_WRITE_HEADER_FN}(start, ${kindTag}u, messageId, ${payloadWords}u);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 0u] = bitcast<u32>(value[0][0]);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 1u] = bitcast<u32>(value[0][1]);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 2u] = bitcast<u32>(value[0][2]);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 3u] = bitcast<u32>(value[0][3]);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 4u] = bitcast<u32>(value[1][0]);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 5u] = bitcast<u32>(value[1][1]);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 6u] = bitcast<u32>(value[1][2]);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 7u] = bitcast<u32>(value[1][3]);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 8u] = bitcast<u32>(value[2][0]);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 9u] = bitcast<u32>(value[2][1]);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 10u] = bitcast<u32>(value[2][2]);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 11u] = bitcast<u32>(value[2][3]);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 12u] = bitcast<u32>(value[3][0]);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 13u] = bitcast<u32>(value[3][1]);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 14u] = bitcast<u32>(value[3][2]);
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + ${DEBUG_RECORD_HEADER_WORDS}u + 15u] = bitcast<u32>(value[3][3]);
}`;
    }
}

function createDebugSupportWgsl(capacityWords: number): string {
    const helpers = Object.entries(DEBUG_FUNCTIONS)
        .map(([name, kind]) =>
            createDebugFunctionHelperWgsl(
                name as SupportedDebugFunctionName,
                kind
            )
        )
        .join('\n\n');

    return `
struct _volten_debug_storage_buffer {
    cursor: atomic<u32>,
    dropped: atomic<u32>,
    data: array<u32>,
};

var<private> ${VOLTEN_DEBUG_ENABLED_NAME}: bool = false;
var<private> ${VOLTEN_DEBUG_GID_NAME}: vec3<u32> = vec3<u32>(0u);

fn ${VOLTEN_DEBUG_BEGIN_FN}(gid: vec3<u32>) {
    ${VOLTEN_DEBUG_ENABLED_NAME} = false;
    ${VOLTEN_DEBUG_GID_NAME} = gid;
}

fn enableDebug() {
    ${VOLTEN_DEBUG_ENABLED_NAME} = true;
}

fn ${VOLTEN_DEBUG_RESERVE_FN}(payloadWords: u32) -> u32 {
    let totalWords = ${DEBUG_RECORD_HEADER_WORDS}u + payloadWords;
    let start = atomicAdd(&${VOLTEN_DEBUG_BUFFER_NAME}.cursor, totalWords);
    if (start + totalWords > ${capacityWords}u) {
        atomicAdd(&${VOLTEN_DEBUG_BUFFER_NAME}.dropped, 1u);
        return ${VOLTEN_DEBUG_OVERFLOW_SENTINEL};
    }
    return start;
}

fn ${VOLTEN_DEBUG_WRITE_HEADER_FN}(start: u32, kind: u32, messageId: u32, payloadWords: u32) {
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + 0u] = kind;
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + 1u] = messageId;
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + 2u] = ${VOLTEN_DEBUG_GID_NAME}.x;
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + 3u] = ${VOLTEN_DEBUG_GID_NAME}.y;
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + 4u] = ${VOLTEN_DEBUG_GID_NAME}.z;
    ${VOLTEN_DEBUG_BUFFER_NAME}.data[start + 5u] = payloadWords;
}

${helpers}`;
}

export function createDebugTransform(
    capacityWords: number
): DebugShaderTransform {
    let messages: readonly string[] = [];

    return {
        transformSource(source) {
            const rewritten = rewriteDebugSource(source);
            messages = rewritten.messages;
            return rewritten.source;
        },
        supportWgsl: [createDebugSupportWgsl(capacityWords)],
        beforeUserMain: (gidName) => `${VOLTEN_DEBUG_BEGIN_FN}(${gidName});`,
        get messages() {
            return messages;
        }
    };
}

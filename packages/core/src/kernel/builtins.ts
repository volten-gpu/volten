import { WGSL_TYPE_ALIASES } from '../types/primitives.js';

/**
 * Builtin shorthand expansion for compute shader entry points.
 *
 * Volten supports convention-based parameter names that automatically
 * expand to @builtin(...) decorators in WGSL:
 *
 * | Shorthand | Type        | Expands To                                      |
 * |-----------|-------------|-------------------------------------------------|
 * | gid       | vec3<u32>   | @builtin(global_invocation_id) gid: vec3<u32>   |
 * | lid       | u32         | @builtin(local_invocation_index) lid: u32       |
 * | wid       | vec3<u32>   | @builtin(workgroup_id) wid: vec3<u32>           |
 * | lid3      | vec3<u32>   | @builtin(local_invocation_id) lid3: vec3<u32>   |
 * | nwg       | vec3<u32>   | @builtin(num_workgroups) nwg: vec3<u32>         |
 */

/**
 * Builtin shorthand definitions.
 * Maps shorthand names to their builtin name and expected type.
 */
export const BUILTIN_SHORTHANDS: Record<
    string,
    { builtin: string; type: string }
> = {
    gid: { builtin: 'global_invocation_id', type: 'vec3<u32>' },
    lid: { builtin: 'local_invocation_index', type: 'u32' },
    wid: { builtin: 'workgroup_id', type: 'vec3<u32>' },
    lid3: { builtin: 'local_invocation_id', type: 'vec3<u32>' },
    nwg: { builtin: 'num_workgroups', type: 'vec3<u32>' }
};

/** Hidden uniform name injected by Volten for guarded dispatch bounds. */
export const VOLTEN_BOUNDS_NAME =
    '_volten_dispatch_bounds_guard_uniform';

const VOLTEN_USER_MAIN_NAME =
    '_volten_user_main_entrypoint_wrapper';
const VOLTEN_GID_NAME = '_volten_guard_gid_builtin';

/** Regex to find the start of the main function signature. */
const MAIN_FN_START_REGEX = /fn\s+main\s*\(/;

interface ParsedParameter {
    readonly original: string;
    readonly name: string;
    readonly type: string;
    readonly builtin?: string;
}

interface MainFunctionMatch {
    readonly signatureStart: number;
    readonly paramsStart: number;
    readonly paramsEnd: number;
    readonly params: string;
}

interface FinalizeComputeEntryPointOptions {
    readonly guarded?: boolean;
    readonly forceWrapper?: boolean;
    readonly requireGlobalInvocationId?: boolean;
    readonly entryPointPrelude?: string | ((gidName: string) => string);
}

/**
 * Normalize a WGSL type string for comparison.
 * Resolves shorthand aliases (e.g., vec3u → vec3<u32>) and strips
 * whitespace so that "vec3< u32 >" matches "vec3<u32>".
 */
function normalizeType(type: string): string {
    const stripped = type.replace(/\s+/g, '').toLowerCase();
    return WGSL_TYPE_ALIASES[stripped] ?? stripped;
}

function findMainFunction(source: string): MainFunctionMatch | null {
    const match = MAIN_FN_START_REGEX.exec(source);
    if (!match) {
        return null;
    }

    const signatureStart = match.index;
    const paramsStart = signatureStart + match[0].length;
    let depth = 1;

    for (let i = paramsStart; i < source.length; i++) {
        const char = source[i];
        if (char === '(') {
            depth++;
        } else if (char === ')') {
            depth--;
            if (depth === 0) {
                return {
                    signatureStart,
                    paramsStart,
                    paramsEnd: i,
                    params: source.slice(paramsStart, i)
                };
            }
        }
    }

    return null;
}

/**
 * Expand builtin shorthands in a single parameter string.
 *
 * @param paramStr - A single parameter like "gid: vec3<u32>"
 * @returns The expanded parameter or original if no shorthand matched
 * @throws Error if shorthand is used with incorrect type
 */
export function expandParameter(paramStr: string): string {
    const trimmed = paramStr.trim();
    if (!trimmed) return trimmed;

    // Already has @builtin decorator - leave it alone
    if (trimmed.startsWith('@builtin')) {
        return trimmed;
    }

    // Parse "name: type"
    const match = trimmed.match(/^([\w]+)\s*:\s*([\w<>]+)$/);
    if (!match) {
        return trimmed; // Not a simple "name: type" pattern
    }

    const [, name, type] = match;
    const shorthand = BUILTIN_SHORTHANDS[name];

    if (!shorthand) {
        return trimmed; // Not a known shorthand
    }

    // Validate type matches expected
    if (normalizeType(type) !== normalizeType(shorthand.type)) {
        throw new Error(
            `Volten Error: Shorthand "${name}" expects type "${shorthand.type}", ` +
                `but got "${type}". Either use the correct type or rename the parameter.`
        );
    }

    // Expand to full @builtin decorator
    return `@builtin(${shorthand.builtin}) ${name}: ${type}`;
}

/**
 * Expand all builtin shorthands in a function's parameter list.
 *
 * @param params - The parameter list string (without parentheses)
 * @returns The expanded parameter list
 */
export function expandParameterList(params: string): string {
    if (!params.trim()) return params;

    return params
        .split(',')
        .map((p) => expandParameter(p))
        .join(', ');
}

/**
 * Expand builtin shorthands in the main function of shader source.
 * Only processes the function named "main" (the compute entry point).
 *
 * @param source - The WGSL shader source
 * @returns The source with expanded shorthands in main's parameters
 */
export function expandBuiltinShorthands(source: string): string {
    const main = findMainFunction(source);
    if (!main) {
        return source;
    }

    const expanded = expandParameterList(main.params);
    return (
        source.slice(0, main.paramsStart) +
        expanded +
        source.slice(main.paramsEnd)
    );
}

function parseParameter(paramStr: string): ParsedParameter {
    const trimmed = paramStr.trim();

    const builtinMatch = trimmed.match(
        /^@builtin\(([\w_]+)\)\s+([\w]+)\s*:\s*([\w<>]+)$/
    );
    if (builtinMatch) {
        const [, builtin, name, type] = builtinMatch;
        return { original: trimmed, builtin, name, type };
    }

    const simpleMatch = trimmed.match(/^([\w]+)\s*:\s*([\w<>]+)$/);
    if (simpleMatch) {
        const [, name, type] = simpleMatch;
        return { original: trimmed, name, type };
    }

    throw new Error(
        `Volten Error: Failed to parse entry-point parameter "${trimmed}".`
    );
}

function parseParameterList(params: string): ParsedParameter[] {
    if (!params.trim()) {
        return [];
    }

    return params.split(',').map((p) => parseParameter(p));
}

function formatComputeDecorators(
    workgroupSize: [number, number, number]
): string {
    const [x, y, z] = workgroupSize;
    return `@compute @workgroup_size(${x}, ${y}, ${z})\n`;
}

/**
 * Finalize the compute entry point after shorthand expansion.
 *
 * In unguarded mode, this simply decorates the user's main function.
 * In guarded mode, it rewrites the user's main into a helper and emits
 * a wrapped compute entry point that clamps over-dispatched invocations.
 */
function finalizeComputeEntryPoint(
    source: string,
    workgroupSize: [number, number, number] = [64, 1, 1],
    options?: FinalizeComputeEntryPointOptions
): string {
    const main = findMainFunction(source);
    if (!main) {
        return source;
    }

    const decorators = formatComputeDecorators(workgroupSize);
    const needsWrapper =
        options?.guarded ||
        options?.forceWrapper ||
        options?.requireGlobalInvocationId ||
        options?.entryPointPrelude;

    if (!needsWrapper) {
        return (
            source.slice(0, main.signatureStart) +
            decorators +
            source.slice(main.signatureStart)
        );
    }

    /*
    parsedParams example:
    [{
        "original": "@builtin(global_invocation_id) gid: vec3u",
        "builtin": "global_invocation_id",
        "name": "gid",
        "type": "vec3u"
    }]
    */
    const parsedParams = parseParameterList(main.params);
    const existingGid = parsedParams.find(
        (param) => param.builtin === 'global_invocation_id'
    );
    const guardGidName = existingGid?.name ?? VOLTEN_GID_NAME;
    const entryParams = parsedParams.map((param) => param.original);
    const needsInjectedGid =
        !existingGid &&
        (options?.guarded ||
            options?.requireGlobalInvocationId ||
            options?.entryPointPrelude);

    if (needsInjectedGid) {
        entryParams.push(
            `@builtin(global_invocation_id) ${VOLTEN_GID_NAME}: vec3<u32>`
        );
    }

    /*
        from: @builtin(global_invocation_id) gid: vec3u
        to: "gid: vec3u"
    */
    const helperParams = parsedParams
        .map((param) => `${param.name}: ${param.type}`)
        .join(', ');

    /*
        renamedSource goes from:
            fn main(gid: vec3u) {
        to:
            fn _volten_user_main_entrypoint_wrapper(gid: vec3u) {
    */
    const renamedSource =
        source.slice(0, main.signatureStart) +
        `fn ${VOLTEN_USER_MAIN_NAME}(${helperParams})` +
        source.slice(main.paramsEnd + 1);

    const userArgs = parsedParams.map((param) => param.name).join(', ');

    // shader part that calls the wrapper function with the
    // provided arguments:
    // e.g.: callLine = "_volten_user_main_entrypoint_wrapper(gid);"
    const callLine =
        userArgs.length > 0
            ? `${VOLTEN_USER_MAIN_NAME}(${userArgs});`
            : `${VOLTEN_USER_MAIN_NAME}();`;

    const guardedEntryPoint = `${decorators}fn main(${entryParams.join(', ')}) {
${options?.guarded ? `    if (any(${guardGidName} >= ${VOLTEN_BOUNDS_NAME}.xyz)) {
        return;
    }` : ''}
${(() => {
        const prelude = options?.entryPointPrelude;
        if (!prelude) {
            return '';
        }

        const resolvedPrelude =
            typeof prelude === 'function' ? prelude(guardGidName) : prelude;
        return resolvedPrelude
            .split('\n')
            .map((line) => `    ${line}`)
            .join('\n');
    })()}
    ${callLine}
}`;

    return `${renamedSource}\n\n${guardedEntryPoint}`;
}

/**
 * Process shader source: expand shorthands and inject decorators.
 * This is the main entry point for shader source transformation.
 *
 * @param source - The WGSL shader source
 * @param workgroupSize - The workgroup size
 * @returns Fully processed shader source
 */
export function processShaderSource(
    source: string,
    workgroupSize: [number, number, number] = [64, 1, 1],
    options?: {
        unsafeManualBounds?: boolean;
        forceWrapper?: boolean;
        requireGlobalInvocationId?: boolean;
        entryPointPrelude?: string | ((gidName: string) => string);
    }
): string {
    const expanded = expandBuiltinShorthands(source);
    return finalizeComputeEntryPoint(expanded, workgroupSize, {
        guarded: !options?.unsafeManualBounds,
        forceWrapper: options?.forceWrapper,
        requireGlobalInvocationId: options?.requireGlobalInvocationId,
        entryPointPrelude: options?.entryPointPrelude
    });
}

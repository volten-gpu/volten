import { WGSL_TYPE_ALIASES } from '../types/primitives.js';
import {
    findFunctionSignature,
    splitTopLevelCommaList,
    type FunctionSignatureMatch
} from './wgsl-source.js';

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

interface ParsedParameter {
    readonly original: string;
    readonly name: string;
    readonly type: string;
    readonly builtin?: string;
}

export type EntryPointSetup = string | ((gidName: string) => string);

interface FinalizeComputeEntryPointOptions {
    readonly boundsGuard?: boolean;
    readonly beforeUserMain?: readonly EntryPointSetup[];
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

function findMainFunction(source: string): FunctionSignatureMatch | null {
    return findFunctionSignature(source, 'main');
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

    return splitTopLevelCommaList(params)
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

    return splitTopLevelCommaList(params).map((p) => parseParameter(p));
}

function formatComputeDecorators(
    workgroupSize: [number, number, number]
): string {
    const [x, y, z] = workgroupSize;
    return `@compute @workgroup_size(${x}, ${y}, ${z})\n`;
}

function indentWgslBlock(source: string): string {
    return source
        .trim()
        .split('\n')
        .map((line) => `    ${line}`)
        .join('\n');
}

function formatBoundsGuard(
    enabled: boolean | undefined,
    gidName: string
): string {
    if (!enabled) {
        return '';
    }

    return `    if (any(${gidName} >= ${VOLTEN_BOUNDS_NAME}.xyz)) {
        return;
    }`;
}

function formatEntryPointSetup(
    setup: readonly EntryPointSetup[],
    gidName: string
): string {
    return setup
        .map((entry) => (typeof entry === 'function' ? entry(gidName) : entry))
        .filter((entry) => entry.trim().length > 0)
        .map((entry) => indentWgslBlock(entry))
        .join('\n');
}

/**
 * Finalize the compute entry point after shorthand expansion.
 *
 * If wrapping is needed, this renames the user's main into a helper and emits
 * the real compute entry point with optional bounds/setup code.
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
    const beforeUserMain = options?.beforeUserMain ?? [];
    const setupNeedsGid = beforeUserMain.some(
        (entry) => typeof entry === 'function'
    );
    const needsWrapper = options?.boundsGuard || beforeUserMain.length > 0;

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
        !existingGid && (options?.boundsGuard || setupNeedsGid);

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

    const guardBlock = formatBoundsGuard(options?.boundsGuard, guardGidName);
    const setupBlock = formatEntryPointSetup(beforeUserMain, guardGidName);
    const entryPoint = [
        `${decorators}fn main(${entryParams.join(', ')}) {`,
        guardBlock,
        setupBlock,
        `    ${callLine}`,
        '}'
    ]
        .filter((section) => section.length > 0)
        .join('\n');

    return `${renamedSource}\n\n${entryPoint}`;
}

/**
 * Finalize kernel WGSL after optional shader transforms have run.
 *
 * @param source - The WGSL shader source
 * @param workgroupSize - The workgroup size
 * @returns Fully processed shader source
 */
export function finalizeKernelSource(
    source: string,
    workgroupSize: [number, number, number] = [64, 1, 1],
    options?: {
        unsafeManualBounds?: boolean;
        beforeUserMain?: readonly EntryPointSetup[];
    }
): string {
    const expanded = expandBuiltinShorthands(source);
    return finalizeComputeEntryPoint(expanded, workgroupSize, {
        boundsGuard: !options?.unsafeManualBounds,
        beforeUserMain: options?.beforeUserMain
    });
}

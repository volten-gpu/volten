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
export const BUILTIN_SHORTHANDS: Record<string, { builtin: string; type: string }> = {
    gid: { builtin: 'global_invocation_id', type: 'vec3<u32>' },
    lid: { builtin: 'local_invocation_index', type: 'u32' },
    wid: { builtin: 'workgroup_id', type: 'vec3<u32>' },
    lid3: { builtin: 'local_invocation_id', type: 'vec3<u32>' },
    nwg: { builtin: 'num_workgroups', type: 'vec3<u32>' },
};

/**
 * WGSL shorthand type aliases.
 * Maps shorthand forms (e.g., vec3u) to their canonical generic forms (e.g., vec3<u32>).
 * Covers vec2/3/4 × u32/i32/f32/f16 and mat variants.
 */
export const WGSL_TYPE_ALIASES: Record<string, string> = {
    // vec2
    vec2u: 'vec2<u32>',
    vec2i: 'vec2<i32>',
    vec2f: 'vec2<f32>',
    vec2h: 'vec2<f16>',
    // vec3
    vec3u: 'vec3<u32>',
    vec3i: 'vec3<i32>',
    vec3f: 'vec3<f32>',
    vec3h: 'vec3<f16>',
    // vec4
    vec4u: 'vec4<u32>',
    vec4i: 'vec4<i32>',
    vec4f: 'vec4<f32>',
    vec4h: 'vec4<f16>',
    // common matrix shorthands
    mat2x2f: 'mat2x2<f32>',
    mat2x3f: 'mat2x3<f32>',
    mat2x4f: 'mat2x4<f32>',
    mat3x2f: 'mat3x2<f32>',
    mat3x3f: 'mat3x3<f32>',
    mat3x4f: 'mat3x4<f32>',
    mat4x2f: 'mat4x2<f32>',
    mat4x3f: 'mat4x3<f32>',
    mat4x4f: 'mat4x4<f32>',
    mat2x2h: 'mat2x2<f16>',
    mat2x3h: 'mat2x3<f16>',
    mat2x4h: 'mat2x4<f16>',
    mat3x2h: 'mat3x2<f16>',
    mat3x3h: 'mat3x3<f16>',
    mat3x4h: 'mat3x4<f16>',
    mat4x2h: 'mat4x2<f16>',
    mat4x3h: 'mat4x3<f16>',
    mat4x4h: 'mat4x4<f16>',
};

/**
 * Regex to match the main function signature.
 * Captures the full "fn main(...)" block including parameters.
 */
const MAIN_FN_REGEX = /fn\s+main\s*\(([^)]*)\)/;

/**
 * Normalize a WGSL type string for comparison.
 * Resolves shorthand aliases (e.g., vec3u → vec3<u32>) and strips
 * whitespace so that "vec3< u32 >" matches "vec3<u32>".
 */
function normalizeType(type: string): string {
    const stripped = type.replace(/\s+/g, '').toLowerCase();
    return WGSL_TYPE_ALIASES[stripped] ?? stripped;
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
        .map(p => expandParameter(p))
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
    return source.replace(MAIN_FN_REGEX, (match, params) => {
        const expanded = expandParameterList(params);
        return `fn main(${expanded})`;
    });
}

/**
 * Inject @compute and @workgroup_size decorators before the main function.
 * 
 * @param source - The WGSL shader source
 * @param workgroupSize - The workgroup size as [x, y, z]
 * @returns The source with compute decorators injected
 */
export function injectComputeDecorators(
    source: string,
    workgroupSize: [number, number, number] = [64, 1, 1]
): string {
    const [x, y, z] = workgroupSize;
    const decorators = `@compute @workgroup_size(${x}, ${y}, ${z})\n`;

    // Insert decorators before "fn main"
    return source.replace(
        /fn\s+main\s*\(/,
        `${decorators}fn main(`
    );
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
    workgroupSize: [number, number, number] = [64, 1, 1]
): string {
    let processed = expandBuiltinShorthands(source);
    processed = injectComputeDecorators(processed, workgroupSize);
    return processed;
}

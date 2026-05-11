export interface FunctionSignatureMatch {
    readonly signatureStart: number;
    readonly paramsStart: number;
    readonly paramsEnd: number;
    readonly params: string;
}

export interface ParenthesizedSpan {
    readonly content: string;
    readonly endIndex: number;
}

export function isIdentifierStart(char: string): boolean {
    return /[A-Za-z_]/.test(char);
}

export function isIdentifierPart(char: string): boolean {
    return /[A-Za-z0-9_]/.test(char);
}

export function skipWhitespace(source: string, index: number): number {
    let current = index;
    while (current < source.length && /\s/.test(source[current])) {
        current++;
    }
    return current;
}

export function findFunctionSignature(
    source: string,
    functionName: string
): FunctionSignatureMatch | null {
    const escapedName = functionName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const match = new RegExp(`\\bfn\\s+${escapedName}\\s*\\(`).exec(source);
    if (!match) {
        return null;
    }

    const signatureStart = match.index;
    const paramsStart = signatureStart + match[0].length;
    const span = parseParenthesizedSpan(source, paramsStart - 1, {
        unterminatedMessage: `Volten Error: Unterminated parameter list for function "${functionName}".`
    });

    return {
        signatureStart,
        paramsStart,
        paramsEnd: span.endIndex - 1,
        params: span.content
    };
}

export function parseParenthesizedSpan(
    source: string,
    openParenIndex: number,
    options?: {
        readonly unterminatedMessage?: string;
    }
): ParenthesizedSpan {
    let depth = 1;
    let index = openParenIndex + 1;
    let quote: '"' | "'" | null = null;
    let lineComment = false;
    let blockComment = false;

    while (index < source.length) {
        const char = source[index];
        const next = source[index + 1];

        if (lineComment) {
            if (char === '\n') {
                lineComment = false;
            }
            index++;
            continue;
        }

        if (blockComment) {
            if (char === '*' && next === '/') {
                blockComment = false;
                index += 2;
                continue;
            }
            index++;
            continue;
        }

        if (quote) {
            if (char === '\\') {
                index += 2;
                continue;
            }
            if (char === quote) {
                quote = null;
            }
            index++;
            continue;
        }

        if (char === '/' && next === '/') {
            lineComment = true;
            index += 2;
            continue;
        }

        if (char === '/' && next === '*') {
            blockComment = true;
            index += 2;
            continue;
        }

        if (char === '"' || char === "'") {
            quote = char;
            index++;
            continue;
        }

        if (char === '(') {
            depth++;
            index++;
            continue;
        }

        if (char === ')') {
            depth--;
            if (depth === 0) {
                return {
                    content: source.slice(openParenIndex + 1, index),
                    endIndex: index + 1
                };
            }
            index++;
            continue;
        }

        index++;
    }

    throw new Error(
        options?.unterminatedMessage ??
            'Volten Error: Unterminated parenthesized expression while parsing WGSL source.'
    );
}

export function splitTopLevelCommaList(source: string): string[] {
    const items: string[] = [];
    let start = 0;
    let depth = 0;
    let index = 0;
    let quote: '"' | "'" | null = null;
    let lineComment = false;
    let blockComment = false;

    while (index < source.length) {
        const char = source[index];
        const next = source[index + 1];

        if (lineComment) {
            if (char === '\n') {
                lineComment = false;
            }
            index++;
            continue;
        }

        if (blockComment) {
            if (char === '*' && next === '/') {
                blockComment = false;
                index += 2;
                continue;
            }
            index++;
            continue;
        }

        if (quote) {
            if (char === '\\') {
                index += 2;
                continue;
            }
            if (char === quote) {
                quote = null;
            }
            index++;
            continue;
        }

        if (char === '/' && next === '/') {
            lineComment = true;
            index += 2;
            continue;
        }

        if (char === '/' && next === '*') {
            blockComment = true;
            index += 2;
            continue;
        }

        if (char === '"' || char === "'") {
            quote = char;
            index++;
            continue;
        }

        if (char === '(' || char === '[' || char === '{') {
            depth++;
            index++;
            continue;
        }

        if (char === ')' || char === ']' || char === '}') {
            depth--;
            index++;
            continue;
        }

        if (char === ',' && depth === 0) {
            items.push(source.slice(start, index).trim());
            start = index + 1;
        }

        index++;
    }

    const last = source.slice(start).trim();
    if (last) {
        items.push(last);
    }

    return items;
}

const labelCounters = new Map<string, number>();

function nextLabelIndex(kind: string): number {
    const next = (labelCounters.get(kind) ?? 0) + 1;
    labelCounters.set(kind, next);
    return next;
}

function normalizeExplicitLabel(label?: string): string | undefined {
    const trimmed = label?.trim();
    return trimmed ? trimmed : undefined;
}

export function makeLabel(kind: string, explicit?: string): string {
    const normalized = normalizeExplicitLabel(explicit);
    if (normalized) {
        return normalized;
    }
    return `${kind}#${nextLabelIndex(kind)}`;
}

export function makeInvocationLabel(
    operationLabel: string,
    explicit?: string
): string {
    const normalized = normalizeExplicitLabel(explicit);
    if (normalized) {
        return normalized;
    }
    return `${operationLabel}::node#${nextLabelIndex('Node')}`;
}

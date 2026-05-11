let nextResourceId = 1;

export function createResourceId(): number {
    return nextResourceId++;
}

import { Buffer } from '../data/buffer.js';
import { RawBuffer } from '../data/raw-buffer.js';
import { Uniform } from '../data/uniform.js';
import type { DebugOptions } from '../debug/types.js';
import { makeInvocationLabel } from '../utils/labels.js';
import type { OperationDefinition } from './operation.js';

/** Per-invocation options shared by primitive kernels and plans. */
export interface InvocationOptions {
    threads?: number | [number] | [number, number] | [number, number, number];
    label?: string;
    debug?: boolean | DebugOptions;
}

export type BindingValue = Buffer | RawBuffer | Uniform | Handle;
export type Bindings = Readonly<Record<string, BindingValue>>;

/** A context-free reference to a binding exposed by a logical node. */
export interface Handle {
    readonly _kind: 'logical-handle';
    readonly _id: symbol;
    readonly _node: Node;
    readonly _name: string;
    readonly _label: string;
}

export interface NodeBase {
    readonly _kind: 'logical-node';
    readonly _id: symbol;
    readonly _label: string;
    readonly _operation: OperationDefinition;
    readonly _bindings: Bindings;
    readonly _options: Readonly<InvocationOptions>;
    _contextOwner?: symbol;
}

export type Node<
    THandles extends Record<string, Handle> = Record<never, Handle>
> = NodeBase & THandles;

export type HandlesForBindings<TBindings extends Bindings> = {
    readonly [K in keyof TBindings as TBindings[K] extends
        | Buffer
        | RawBuffer
        | Handle
        ? K
        : never]: Handle;
};

export function isHandle(value: unknown): value is Handle {
    return (
        typeof value === 'object' &&
        value !== null &&
        (value as Partial<Handle>)._kind === 'logical-handle'
    );
}

export function isNode(value: unknown): value is Node {
    return (
        typeof value === 'object' &&
        value !== null &&
        (value as Partial<NodeBase>)._kind === 'logical-node'
    );
}

function isBufferLike(value: BindingValue): boolean {
    return (
        value instanceof Buffer || value instanceof RawBuffer || isHandle(value)
    );
}

function createHandle(node: Node, name: string): Handle {
    const handle: Handle = Object.freeze({
        _kind: 'logical-handle',
        _id: Symbol(`Handle:${name}`),
        _node: node,
        _name: name,
        _label: `${node._label}.${name}`
    });

    Object.defineProperty(node, name, {
        value: handle,
        enumerable: true,
        configurable: false
    });
    return handle;
}

/**
 * Creates the context-free node returned by a callable operation.
 *
 * Buffer-like *input* handles are installed eagerly to preserve Volten's
 * pass-through chaining model.
 *
 * e.g.
 *
 * const A = reduce({ input });
 * A.input; // already exists
 *
 * Other *output* handles are created on first
 * property access by a Proxy. This lets a plan expose the keys returned by build()
 * without executing that context-dependent function during graph authoring.
 */
export function createLogicalNode(
    operation: OperationDefinition,
    bindings: Bindings,
    options: InvocationOptions = {}
): Node {
    const target: NodeBase = {
        _kind: 'logical-node',
        _id: Symbol('LogicalNode'),
        _label: makeInvocationLabel(operation.label, options.label),
        _operation: operation,
        _bindings: Object.freeze({ ...bindings }),
        _options: Object.freeze({ ...options })
    };

    const node = new Proxy(target as Node, {
        get(current, property, receiver) {
            if (Reflect.has(current, property)) {
                return Reflect.get(current, property, receiver);
            }
            if (typeof property !== 'string' || property === 'then') {
                return undefined;
            }
            if (property.startsWith('_')) {
                return undefined;
            }
            if (operation._kind === 'kernel') {
                if (!operation.outputNames?.includes(property))
                    return undefined;
            }
            return createHandle(receiver as Node, property);
        }
    });

    for (const [name, value] of Object.entries(bindings)) {
        if (name.startsWith('_')) {
            throw new Error(
                `Volten Error: Binding name "${name}" cannot start with "_".`
            );
        }
        if (isBufferLike(value)) createHandle(node, name);
    }

    return node;
}

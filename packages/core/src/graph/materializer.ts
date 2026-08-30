import { Buffer } from '../data/buffer.js';
import { RawBuffer } from '../data/raw-buffer.js';
import { Uniform } from '../data/uniform.js';
import {
    DebugBufferResource,
    VOLTEN_DEBUG_BUFFER_NAME,
    createDebugTransform,
    resolveDebugOptions
} from '../debug/index.js';
import {
    assembleFullShader,
    generateBindings,
    resolveBounds,
    resolveDispatch
} from '../kernel/bindings.js';
import { VOLTEN_BOUNDS_NAME } from '../kernel/builtins.js';
import { isKernelOperation } from '../kernel/kernel.js';
import { prepareKernelShader } from '../kernel/shader.js';
import { isPlanOperation } from '../plan/plan.js';
import type { UniformLayoutMode } from '../utils/uniform-layout.js';
import {
    createDispatchNode,
    getDispatchNodeHandles,
    type DispatchHandle,
    type DispatchNode
} from './dispatch-node.js';
import type { MaterializedInvocation } from './compiler.js';
import { isHandle, type BindingValue, type Handle, type Node } from './node.js';
import type { OperationContext } from './operation.js';
import type { PipelineCache } from './pipeline-cache.js';

export interface MaterializerHost {
    readonly device: GPUDevice;
    readonly pipelineCache: PipelineCache;
    readonly uniformLayoutMode: UniformLayoutMode;
}

export interface ResolvedHandle {
    readonly resource: Buffer | RawBuffer | DispatchHandle;
    readonly completion: readonly DispatchNode[];
}

export interface LoweredNode {
    readonly outputs: ReadonlyMap<string, ResolvedHandle>;
    readonly outputNames: readonly string[];
    readonly terminals: readonly DispatchNode[];
}

/**
 * Lowers context-free operation nodes into the physical dispatch graph.
 *
 * A resolved handle intentionally contains two independent facts:
 *
 * - resource: the buffer supplied to a downstream shader binding
 * - completion: the dispatches that must finish before that binding is used
 *
 * For a primitive kernel these usually identify the same producer. For a
 * multi-branch plan, one result buffer may come from one branch while the
 * completion list contains every branch returned by the plan
 */
export class Materializer {
    private readonly contextToken = Symbol('VoltenContext');
    private readonly context: OperationContext;
    private readonly cache = new WeakMap<Node, LoweredNode>();
    private readonly active = new WeakSet<Node>();

    constructor(private readonly host: MaterializerHost) {
        this.context = Object.freeze({
            device: host.device,
            features: host.device.features,
            limits: host.device.limits
        });
    }

    materialize(nodes: readonly Node[]): MaterializedInvocation[] {
        return nodes.map((node) => ({
            terminals: this.lowerNode(node).terminals
        }));
    }

    getCached(node: Node): LoweredNode | undefined {
        return this.cache.get(node);
    }

    lowerNode(node: Node): LoweredNode {
        this.claim(node);

        const cached = this.cache.get(node);
        if (cached) return cached;
        if (this.active.has(node)) {
            throw new Error(
                `Volten Error: Cycle detected while building "${node._label}".`
            );
        }

        this.active.add(node);
        try {
            const lowered = isKernelOperation(node._operation)
                ? this.lowerKernel(node)
                : this.lowerPlan(node);
            this.cache.set(node, lowered);
            return lowered;
        } finally {
            this.active.delete(node);
        }
    }

    resolveHandle(handle: Handle): ResolvedHandle {
        const lowered = this.lowerNode(handle._node);
        const resolved = lowered.outputs.get(handle._name);
        if (!resolved) {
            throw new Error(
                `Volten Error: Operation "${handle._node._label}" does not expose output "${handle._name}".`
            );
        }
        return resolved;
    }

    private lowerKernel(node: Node): LoweredNode {
        const operation = node._operation;
        if (!isKernelOperation(operation)) {
            throw new Error('Volten Error: Expected a kernel operation.');
        }

        const kernel = operation._definition.resolve(this.context);
        const dependencies = new Set<DispatchNode>();
        const bindings: Record<string, unknown> = {};

        for (const [name, value] of Object.entries(node._bindings)) {
            if (isHandle(value)) {
                const resolved = this.resolveHandle(value);
                bindings[name] = resolved.resource;
                for (const dependency of resolved.completion) {
                    dependencies.add(dependency);
                }
            } else {
                bindings[name] = value;
            }
        }

        this.validateReservedBindings(bindings);
        const bounds = resolveBounds(kernel, bindings, node._options.threads);
        const dispatch = resolveDispatch(
            kernel,
            bindings,
            node._options.threads
        );

        if (
            !kernel.unsafeManualBounds &&
            kernel.usesBarrier &&
            dispatchNeedsBoundsGuard(bounds, dispatch, kernel.workgroupSize)
        ) {
            throw new Error(
                'Volten Error: This kernel uses workgroup/storage barriers and requires a guarded partial workgroup.\n' +
                    '  Make threads an exact multiple of workgroupSize, or set unsafeManualBounds: true and handle bounds in WGSL.'
            );
        }

        const ownedResources = [];
        const executionBindings = { ...bindings };

        if (!kernel.unsafeManualBounds) {
            const boundsUniform = new Uniform(
                [bounds[0], bounds[1], bounds[2], 0],
                'vec4u',
                { label: `${node._label} bounds` }
            );
            ownedResources.push(boundsUniform);
            executionBindings[VOLTEN_BOUNDS_NAME] = boundsUniform;
        }

        const debugOptions = resolveDebugOptions(node._options.debug);
        let debugResource: DebugBufferResource | null = null;
        let debugTransform: ReturnType<typeof createDebugTransform> | null =
            null;

        if (debugOptions) {
            debugTransform = createDebugTransform(debugOptions.capacityWords);
            debugResource = new DebugBufferResource(
                debugOptions,
                `${node._label} debug`
            );
            ownedResources.push(debugResource);
            executionBindings[VOLTEN_DEBUG_BUFFER_NAME] = debugResource.buffer;
        }

        const bindingEntries = generateBindings(executionBindings, {
            uniformLayoutMode: this.host.uniformLayoutMode
        });
        const preparedShader = prepareKernelShader(kernel, {
            transforms: debugTransform ? [debugTransform] : []
        });
        const shaderCode = assembleFullShader(bindingEntries, {
            uniformLayoutMode: this.host.uniformLayoutMode,
            kernelSource: preparedShader.kernelSource,
            additionalSections: preparedShader.supportWgsl
        });
        const { pipeline, bindGroupLayout } =
            this.host.pipelineCache.getOrCreate(
                this.host.device,
                shaderCode,
                kernel.label
            );

        const dispatchNode = createDispatchNode({
            kernel,
            pipeline,
            bindGroupLayout,
            bindingEntries,
            ownedResources,
            bounds: [...bounds],
            dispatch: [...dispatch],
            bindings,
            shaderCode,
            dependencies: [...dependencies],
            label: node._label,
            debug:
                debugResource && debugTransform
                    ? {
                          resource: debugResource,
                          messages: debugTransform.messages
                      }
                    : null
        });

        const outputs = new Map<string, ResolvedHandle>();
        for (const [name, handle] of Object.entries(
            getDispatchNodeHandles(dispatchNode)
        )) {
            outputs.set(name, {
                resource: handle,
                completion: [dispatchNode]
            });
        }

        return {
            outputs,
            outputNames: kernel.outputNames,
            terminals: [dispatchNode]
        };
    }

    private lowerPlan(node: Node): LoweredNode {
        const operation = node._operation;
        if (!isPlanOperation(operation)) {
            throw new Error('Volten Error: Expected a plan operation.');
        }
        if (node._options.debug) {
            throw new Error(
                'Volten Error: Debugging an entire plan is not supported yet. Enable debugging on an internal kernel instead.'
            );
        }

        const built = operation._builder(this.context, node._bindings);
        const returned = Object.entries(built);
        const explicit = new Map<string, ResolvedHandle>();
        const terminals = new Set<DispatchNode>();

        for (const [name, handle] of returned) {
            if (!isHandle(handle)) {
                throw new Error(
                    `Volten Error: Plan "${node._label}" returned "${name}", but it is not a Volten handle.`
                );
            }
            const resolved = this.resolveHandle(handle);
            explicit.set(name, resolved);
            for (const terminal of resolved.completion) terminals.add(terminal);
        }

        const completion = [...terminals];
        const outputs = new Map<string, ResolvedHandle>();

        // Preserve current Volten behavior: every buffer-like input remains
        // available as a public handle. Explicit build results with the same
        // name override this pass-through mapping.
        for (const [name, value] of Object.entries(node._bindings)) {
            const resolved = this.resolveBinding(value);
            if (resolved) {
                outputs.set(name, {
                    resource: resolved.resource,
                    completion: mergeNodes(resolved.completion, completion)
                });
            }
        }

        for (const [name, resolved] of explicit) {
            outputs.set(name, {
                resource: resolved.resource,
                completion
            });
        }

        return {
            outputs,
            outputNames: returned.map(([name]) => name),
            terminals: completion
        };
    }

    private resolveBinding(value: BindingValue): ResolvedHandle | null {
        if (isHandle(value)) return this.resolveHandle(value);
        if (value instanceof Buffer || value instanceof RawBuffer) {
            return { resource: value, completion: [] };
        }
        return null;
    }

    private claim(node: Node): void {
        if (node._contextOwner && node._contextOwner !== this.contextToken) {
            throw new Error(
                `Volten Error: Operation node "${node._label}" has already been materialized by another Volten context.`
            );
        }
        node._contextOwner = this.contextToken;
    }

    private validateReservedBindings(bindings: Record<string, unknown>): void {
        for (const name of [VOLTEN_BOUNDS_NAME, VOLTEN_DEBUG_BUFFER_NAME]) {
            if (name in bindings) {
                throw new Error(
                    `Volten Error: Binding name "${name}" is reserved for internal use.`
                );
            }
        }
    }
}

function mergeNodes(
    first: readonly DispatchNode[],
    second: readonly DispatchNode[]
): DispatchNode[] {
    return [...new Set([...first, ...second])];
}

function dispatchNeedsBoundsGuard(
    bounds: [number, number, number],
    dispatch: [number, number, number],
    workgroupSize: [number, number, number]
): boolean {
    return (
        dispatch[0] * workgroupSize[0] !== bounds[0] ||
        dispatch[1] * workgroupSize[1] !== bounds[1] ||
        dispatch[2] * workgroupSize[2] !== bounds[2]
    );
}

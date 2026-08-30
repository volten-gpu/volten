import {
    createLogicalNode,
    type Bindings,
    type Handle,
    type HandlesForBindings,
    type InvocationOptions,
    type Node
} from '../graph/node.js';
import type {
    OperationContext,
    OperationDefinition
} from '../graph/operation.js';
import { makeLabel } from '../utils/labels.js';

export type PlanOutputs = Readonly<Record<string, Handle>>;
export type PlanBuilder<
    TInputs extends Bindings = Bindings,
    TOutputs extends PlanOutputs = PlanOutputs
> = (context: OperationContext, inputs: TInputs) => TOutputs;

export interface PlanOperation<
    TInputs extends Bindings = Bindings,
    TOutputs extends PlanOutputs = PlanOutputs
> extends OperationDefinition {
    readonly _kind: 'plan';
    readonly _builder: PlanBuilder<TInputs, TOutputs>;

    <TBindings extends TInputs>(
        bindings: TBindings,
        options?: InvocationOptions
    ): Node<
        HandlesForBindings<TBindings> & {
            readonly [K in keyof TOutputs]: Handle;
        }
    >;
}

/**
 * Defines a callable operation that may lower into any number of kernels.
 * Output names are inferred from the handles returned by builder. At runtime,
 * logical nodes create those handles lazily and validate them when built.
 */
export function plan<TInputs extends Bindings, TOutputs extends PlanOutputs>(
    builder: PlanBuilder<TInputs, TOutputs>,
    label?: string
): PlanOperation<TInputs, TOutputs> {
    const operationLabel = makeLabel('Plan', label);

    const operation = ((bindings: TInputs, options?: InvocationOptions) =>
        createLogicalNode(operation, bindings, options)) as PlanOperation<
        TInputs,
        TOutputs
    >;

    Object.defineProperties(operation, {
        _kind: { value: 'plan' },
        _builder: { value: builder },
        label: { value: operationLabel }
    });

    return operation;
}

export function isPlanOperation(
    operation: OperationDefinition
): operation is PlanOperation {
    return operation._kind === 'plan';
}

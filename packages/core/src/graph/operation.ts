/** Context information available while an operation is being materialized. */
export interface OperationContext {
    readonly device: GPUDevice;
    readonly features: GPUSupportedFeatures;
    readonly limits: GPUSupportedLimits;
}

/** Common runtime identity shared by kernel() and plan() callables. */
export interface OperationDefinition {
    readonly _kind: 'kernel' | 'plan';
    readonly label: string;
    readonly outputNames?: readonly string[];
}

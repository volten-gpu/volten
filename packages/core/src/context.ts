/**
 * Options for creating a VoltenContext
 */
export interface VoltenOptions {
    /** Pre-existing GPUDevice to use. If not provided, one will be requested. */
    device?: GPUDevice;

    /** Pre-existing GPUAdapter to use. Only used if device is not provided. */
    adapter?: GPUAdapter;

    /** Label for debug purposes */
    label?: string;
}

/**
 * The main Volten context - the "v" instance
 * 
 * This is created via the volten() factory function.
 * Contains methods for scheduling GPU compute work.
 */
export class VoltenContext {
    readonly device: GPUDevice;
    readonly label?: string;

    constructor(device: GPUDevice, options?: { label?: string }) {
        this.device = device;
        this.label = options?.label;
    }

    /**
     * Create a compute pass node
     * 
     * @param kernel - The kernel to execute
     * @param bindings - Input/output bindings
     * @param options - Optional pass configuration
     * @returns A node handle for chaining or execution
     */
    pass(
        kernel: unknown,
        bindings?: Record<string, unknown>,
        options?: Record<string, unknown>
    ): unknown {
        // TODO: Implement in next phase
        throw new Error('v.pass() not yet implemented');
    }

    /**
     * Execute a node or chain of nodes (fire and forget)
     * Does not wait for GPU completion.
     * 
     * @param node - The node to execute
     */
    run(node: unknown): void {
        // TODO: Implement in next phase
        throw new Error('v.run() not yet implemented');
    }

    /**
     * Execute a node and wait for GPU completion
     * 
     * @param node - The node to execute
     */
    async wait(node: unknown): Promise<void> {
        // TODO: Implement in next phase
        throw new Error('v.wait() not yet implemented');
    }

    /**
     * Execute a node and read back results to CPU
     * 
     * @param node - The node to read from
     * @returns The CPU-readable data
     */
    async read(node: unknown): Promise<unknown> {
        // TODO: Implement in next phase
        throw new Error('v.read() not yet implemented');
    }
}

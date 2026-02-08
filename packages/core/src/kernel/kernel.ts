// Kernel class
// Stores shader source code and output declarations

/**
 * Kernel class for compute shader definitions
 * 
 * @example
 * const MyKernel = new Kernel(`
 *   fn compute(instanceIndex: u32) {
 *     outputBuffer[instanceIndex] = inputBuffer[instanceIndex] * multiplier;
 *   }
 * `, { outputs: ['outputBuffer'] });
 */
export class Kernel {
    /** The WGSL shader source (user-provided fn body) */
    readonly source: string;

    /** Names of output bindings */
    readonly outputs: string[];

    constructor(
        source: string,
        options?: {
            outputs?: string[];
        }
    ) {
        this.source = source;
        this.outputs = options?.outputs ?? [];
    }
}

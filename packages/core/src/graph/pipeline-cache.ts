// Pipeline cache for compute shader pipelines
// Caches pipelines by shader source to avoid redundant compilations

/**
 * Cached pipeline entry
 */
export interface CachedPipeline {
    readonly pipeline: GPUComputePipeline;
    readonly bindGroupLayout: GPUBindGroupLayout;
}

/**
 * Pipeline cache that lives on the VoltenContext.
 * 
 * Cache key is the full assembled shader source string. Since identical
 * shader code implies identical bind group layout, this is sufficient
 * for correct caching.
 * 
 * @example
 * ```ts
 * const cache = new PipelineCache();
 * const { pipeline, bindGroupLayout } = cache.getOrCreate(device, shaderCode);
 * ```
 */
export class PipelineCache {
    private readonly cache = new Map<string, CachedPipeline>();

    /**
     * Get an existing pipeline or create a new one.
     * 
     * @param device - GPUDevice for pipeline creation
     * @param shaderCode - Full assembled WGSL source (bindings + kernel)
     * @returns Pipeline and bind group layout
     */
    getOrCreate(
        device: GPUDevice,
        shaderCode: string,
        label?: string
    ): CachedPipeline {
        const existing = this.cache.get(shaderCode);
        if (existing) {
            return existing;
        }

        const shaderModule = device.createShaderModule({
            code: shaderCode,
            label: label ? `${label} shader` : undefined
        });

        // Use 'auto' layout — WebGPU derives the bind group layout from the shader
        const pipeline = device.createComputePipeline({
            label,
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });

        const bindGroupLayout = pipeline.getBindGroupLayout(0);

        const entry: CachedPipeline = { pipeline, bindGroupLayout };
        this.cache.set(shaderCode, entry);

        return entry;
    }

    /**
     * Number of cached pipelines (for debugging/testing)
     */
    get size(): number {
        return this.cache.size;
    }

    /**
     * Clear all cached pipelines
     */
    clear(): void {
        this.cache.clear();
    }
}

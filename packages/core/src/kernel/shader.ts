import {
    finalizeKernelSource,
    type EntryPointSetup
} from './builtins.js';
import type { Kernel } from './kernel.js';

export interface ShaderTransform {
    readonly transformSource?: (source: string) => string;
    readonly beforeUserMain?: EntryPointSetup | readonly EntryPointSetup[];
    readonly supportWgsl?: readonly string[];
}

export interface KernelShaderPreparation {
    readonly transforms?: readonly ShaderTransform[];
}

export interface PreparedKernelShader {
    readonly kernelSource: string;
    readonly supportWgsl: string[];
}

function collectEntryPointSetups(
    transforms: readonly ShaderTransform[]
): EntryPointSetup[] {
    return transforms.flatMap((transform) => {
        const setup = transform.beforeUserMain;
        if (!setup) {
            return [];
        }
        return Array.isArray(setup) ? [...setup] : [setup];
    });
}

/**
 * Prepare executable kernel WGSL before binding declarations are added.
 */
export function prepareKernelShader(
    kernel: Kernel,
    preparation: KernelShaderPreparation = {}
): PreparedKernelShader {
    const transforms = preparation.transforms ?? [];
    let source = kernel.source;
    for (const transform of transforms) {
        if (transform.transformSource) {
            source = transform.transformSource(source);
        }
    }
    const beforeUserMain = collectEntryPointSetups(transforms);
    const supportWgsl = transforms.flatMap(
        (transform) => transform.supportWgsl ?? []
    );

    const kernelSource = finalizeKernelSource(source, kernel.workgroupSize, {
        unsafeManualBounds: kernel.unsafeManualBounds,
        beforeUserMain
    });

    return {
        kernelSource,
        supportWgsl: supportWgsl.filter((section) => section.trim().length > 0)
    };
}

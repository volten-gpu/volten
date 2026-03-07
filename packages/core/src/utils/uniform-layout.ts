import type { LayoutRules } from './layout.js';

export type UniformLayoutPreference = 'auto' | 'classic' | 'standard';
export type UniformLayoutMode = 'classic' | 'standard';

export const UNIFORM_BUFFER_STANDARD_LAYOUT_EXTENSION =
    'uniform_buffer_standard_layout';

/**
 * Checks if the current runtime exposes the WGSL extension that allows
 * storage-like layout rules in the uniform address space.
 */
export function supportsUniformBufferStandardLayout(): boolean {
    const nav = (globalThis as { navigator?: unknown }).navigator as
        | {
              gpu?: {
                  wgslLanguageFeatures?: {
                      has?: (feature: string) => boolean;
                  };
              };
          }
        | undefined;

    const hasFeature = nav?.gpu?.wgslLanguageFeatures?.has;
    if (typeof hasFeature !== 'function') {
        return false;
    }

    try {
        return hasFeature(UNIFORM_BUFFER_STANDARD_LAYOUT_EXTENSION);
    } catch {
        return false;
    }
}

/**
 * Resolves the effective uniform layout mode.
 *
 * - auto: standard if available, otherwise classic.
 * - classic: always classic rules.
 * - standard: requires extension support.
 */
export function resolveUniformLayoutMode(
    preference: UniformLayoutPreference = 'auto'
): UniformLayoutMode {
    if (preference === 'classic') {
        return 'classic';
    }

    if (preference === 'standard') {
        if (!supportsUniformBufferStandardLayout()) {
            throw new Error(
                'Volten Error: uniform layout mode "standard" requires WGSL extension ' +
                    `"${UNIFORM_BUFFER_STANDARD_LAYOUT_EXTENSION}", but it is not available in this runtime.`
            );
        }
        return 'standard';
    }

    return supportsUniformBufferStandardLayout() ? 'standard' : 'classic';
}

/**
 * Maps the chosen uniform layout mode to the concrete packing rules the CPU
 * packer should follow for a `var<uniform>` binding.
 *
 * - classic: classic uniform rules
 * - standard: storage-like rules enabled by the WGSL extension
 */
export function getUniformPackingLayoutRules(
    mode: UniformLayoutMode
): LayoutRules {
    return mode === 'standard' ? 'storage' : 'uniform-classic';
}

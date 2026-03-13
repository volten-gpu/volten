type WgslLanguageFeatures = {
    has?: (feature: string) => boolean;
};

type NavigatorWithGpu = {
    gpu?: {
        wgslLanguageFeatures?: WgslLanguageFeatures;
    };
};

/**
 * Checks if the current runtime exposes a given WGSL language feature.
 */
export function supportsWgslLanguageFeature(feature: string): boolean {
    const nav = (globalThis as { navigator?: unknown }).navigator as
        | NavigatorWithGpu
        | undefined;
    const features = nav?.gpu?.wgslLanguageFeatures;

    if (typeof features?.has !== 'function') {
        return false;
    }

    try {
        return features.has(feature);
    } catch {
        return false;
    }
}

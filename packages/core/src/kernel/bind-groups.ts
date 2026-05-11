import type { Node } from '../graph/node.js';
import { resolveBindableResource } from './resource-resolution.js';

export function getOrCreateNodeBindGroup(
    device: GPUDevice,
    node: Node
): GPUBindGroup {
    const entries: GPUBindGroupEntry[] = [];
    const cacheParts: string[] = [];

    for (const entry of node._bindingEntries) {
        const resource = resolveBindableResource(entry.source);
        const gpuBuffer = resource.ensure(device);

        cacheParts.push(
            `${entry.index}:${resource._resourceId}:${resource._gpuResourceVersion}`
        );
        entries.push({
            binding: entry.index,
            resource: {
                buffer: gpuBuffer
            }
        });
    }

    const cacheKey = cacheParts.join('|');
    const cached = node._cachedBindGroup;
    if (cached && cached.key === cacheKey) {
        return cached.bindGroup;
    }

    const bindGroup = device.createBindGroup({
        label: `${node._label} bind group`,
        layout: node._bindGroupLayout,
        entries
    });

    node._cachedBindGroup = {
        key: cacheKey,
        bindGroup
    };

    return bindGroup;
}

import { VoltenContext, type VoltenOptions } from './context.js';

/**
 * Create a Volten context for GPU compute operations
 * 
 * @param options - Configuration options
 * @returns A VoltenContext instance (the "v" object)
 * 
 * @example
 * // Let Volten request the device
 * const v = await volten();
 * 
 * // Use an existing device
 * const v = await volten({ device: myDevice });
 * 
 * // Use an existing adapter
 * const v = await volten({ adapter: myAdapter });
 */
export async function volten(options?: VoltenOptions): Promise<VoltenContext> {
    let device = options?.device;

    if (!device) {
        let adapter = options?.adapter;

        if (!adapter) {
            adapter = await navigator.gpu?.requestAdapter() ?? undefined;

            if (!adapter) {
                throw new Error(
                    'WebGPU is not supported in this environment. ' +
                    'Make sure you are running in a browser with WebGPU support.'
                );
            }
        }

        device = await adapter.requestDevice();

        if (!device) {
            throw new Error('Failed to request WebGPU device from adapter.');
        }
    }

    device.lost.then((info) => {
        console.error(`Volten: WebGPU device was lost: ${info.message}`);
        if (info.reason !== 'destroyed') {
            console.error('Volten: This was unexpected. GPU work may have been interrupted.');
        }
    });

    return new VoltenContext(device, { label: options?.label });
}

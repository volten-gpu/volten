/**
 * E2E: resource features that need a real GPU queue and shader module.
 */
import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:5174';

test.describe('Resource features', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForFunction(
            () => (window as any).__gpuReady === true,
            null,
            { timeout: 10_000 }
        );
    });

    test('Uniform bindings affect compute output and can be updated', async ({
        page
    }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, Uniform, Kernel } = (window as any).Volten;
            const v = await volten();

            const input = new Buffer([1, 2, 3, 4], 'f32');
            const output = new Buffer([0, 0, 0, 0], 'f32', 'rw');
            const scale = new Uniform(2.0, 'f32');
            const kernel = new Kernel(
                `
              fn main(gid: vec3u) {
                output[gid.x] = input[gid.x] * scale;
              }
            `,
                {
                    outputs: ['output'],
                    threads: 'input'
                }
            );

            const node = v.pass(kernel, { input, output, scale });

            await v.wait(node);
            const first = Array.from(await v.read(output));

            scale.set(3.0);
            await v.wait(node);
            const second = Array.from(await v.read(output));

            return { first, second };
        });

        expect(result.first).toEqual([2, 4, 6, 8]);
        expect(result.second).toEqual([3, 6, 9, 12]);
    });

    test('RawBuffer can be used as a storage buffer and read back', async ({
        page
    }) => {
        const result = await page.evaluate(async () => {
            const { volten, RawBuffer, Kernel } = (window as any).Volten;
            const v = await volten();

            const raw = new RawBuffer(
                new Uint32Array([1, 2, 3, 4]).buffer,
                'array<u32>',
                'rw'
            );
            const kernel = new Kernel(
                `
              fn main(gid: vec3u) {
                data[gid.x] = data[gid.x] + 10u;
              }
            `,
                { threads: 4 }
            );

            const node = v.pass(kernel, { data: raw });
            await v.wait(node);

            const output = await v.read(raw);
            return Array.from(new Uint32Array(output));
        });

        expect(result).toEqual([11, 12, 13, 14]);
    });

    test('readDebug decodes logs written by a real shader', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, Uniform, Kernel } = (window as any).Volten;
            const v = await volten();

            const input = new Buffer([1, 2, 3, 4], 'f32', 'rw');
            const scale = new Uniform(2.0, 'f32');
            const kernel = new Kernel(
                `
              fn main(gid: vec3u) {
                if (gid.x == 2u) {
                  enableDebug();
                  debugF32("scaled", input[gid.x] * scale);
                  debugVec3("marker", vec3f(f32(gid.x), scale, 9.0));
                }
              }
            `,
                { threads: 'input' }
            );

            const node = v.pass(
                kernel,
                { input, scale },
                { debug: { bufferSize: 256 } }
            );

            await v.wait(node);
            const debug = await v.readDebug(node);

            return {
                dropped: debug.dropped,
                truncated: debug.truncated,
                logs: debug.logs.map((log: any) => ({
                    kind: log.kind,
                    gid: [...log.gid],
                    message: log.message,
                    value: Array.isArray(log.value)
                        ? [...log.value]
                        : log.value
                }))
            };
        });

        expect(result.dropped).toBe(0);
        expect(result.truncated).toBe(false);
        expect(result.logs).toEqual([
            {
                kind: 'f32',
                gid: [2, 0, 0],
                message: 'scaled',
                value: 6
            },
            {
                kind: 'vec3f',
                gid: [2, 0, 0],
                message: 'marker',
                value: [2, 2, 9]
            }
        ]);
    });
});

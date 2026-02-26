/**
 * E2E: Compute Correctness Tests
 *
 * Verifies that actual GPU compute operations produce numerically correct results.
 * These are the tests that vitest with mocked GPUDevice fundamentally cannot do.
 */
import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:5174';

test.describe('Arithmetic compute operations', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForFunction(() => (window as any).__gpuReady === true, null, { timeout: 10_000 });
    });

    test('multiply each element by 2', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, Kernel } = (window as any).Volten;
            const v = await volten();

            const buf = new Buffer([1, 2, 3, 4, 5], 'f32', 'rw');
            const k = new Kernel(`
              fn main(gid: vec3u) {
                inout[gid.x] = inout[gid.x] * 2.0;
              }
            `);
            const node = v.pass(k, { inout: buf });
            v.run(node);
            const output = await v.read(node);
            return Array.from(output.inout);
        });

        expect(result).toEqual([2, 4, 6, 8, 10]);
    });

    test('add constant to each element', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, Kernel } = (window as any).Volten;
            const v = await volten();

            const buf = new Buffer([10, 20, 30], 'f32', 'rw');
            const k = new Kernel(`
              fn main(gid: vec3u) {
                inout[gid.x] = inout[gid.x] + 100.0;
              }
            `);
            const node = v.pass(k, { inout: buf });
            v.run(node);
            const output = await v.read(node);
            return Array.from(output.inout);
        });

        expect(result).toEqual([110, 120, 130]);
    });

    test('input → output copy kernel (separate buffers)', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, Kernel } = (window as any).Volten;
            const v = await volten();

            const input = new Buffer([5, 10, 15, 20], 'f32');
            const output = new Buffer([0, 0, 0, 0], 'f32', 'rw');
            const k = new Kernel(`
              fn main(gid: vec3u) {
                output[gid.x] = input[gid.x] * 3.0;
              }
            `, {
                outputs: { output: { definedBy: 'input' } },
                threads: 'input',
            });
            const node = v.pass(k, { input, output });
            v.run(node);
            const res = await v.read(node);
            return Array.from(res.output);
        });

        expect(result).toEqual([15, 30, 45, 60]);
    });

    test('integer arithmetic with u32 buffers', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, Kernel } = (window as any).Volten;
            const v = await volten();

            const buf = new Buffer([1, 2, 3, 4], 'u32', 'rw');
            const k = new Kernel(`
              fn main(gid: vec3u) {
                inout[gid.x] = inout[gid.x] + 10u;
              }
            `);
            const node = v.pass(k, { inout: buf });
            v.run(node);
            const output = await v.read(node);
            return Array.from(output.inout);
        });

        expect(result).toEqual([11, 12, 13, 14]);
    });

    test('square each element', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, Kernel } = (window as any).Volten;
            const v = await volten();

            const buf = new Buffer([2, 3, 4, 5], 'f32', 'rw');
            const k = new Kernel(`
              fn main(gid: vec3u) {
                let val = inout[gid.x];
                inout[gid.x] = val * val;
              }
            `);
            const node = v.pass(k, { inout: buf });
            v.run(node);
            const output = await v.read(node);
            return Array.from(output.inout);
        });

        expect(result).toEqual([4, 9, 16, 25]);
    });
});

test.describe('Thread dispatch inference', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForFunction(() => (window as any).__gpuReady === true, null, { timeout: 10_000 });
    });

    test('threads inferred from buffer size processes all elements', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, Kernel } = (window as any).Volten;
            const v = await volten();

            // 128 elements — should auto-infer 128 threads
            const data = Array.from({ length: 128 }, (_, i) => i);
            const buf = new Buffer(data, 'f32', 'rw');
            const k = new Kernel(`
              fn main(gid: vec3u) {
                inout[gid.x] = inout[gid.x] + 1.0;
              }
            `);
            const node = v.pass(k, { inout: buf });
            v.run(node);
            const output = await v.read(node);
            const arr = Array.from(output.inout);
            return {
                length: arr.length,
                first5: arr.slice(0, 5),
                last5: arr.slice(-5),
            };
        });

        expect(result.length).toBe(128);
        expect(result.first5).toEqual([1, 2, 3, 4, 5]);
        expect(result.last5).toEqual([124, 125, 126, 127, 128]);
    });

    test('explicit threads: number overrides buffer size', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, Kernel } = (window as any).Volten;
            const v = await volten();

            // Buffer has 8 elements, but we only process the first 4
            const buf = new Buffer([1, 2, 3, 4, 100, 100, 100, 100], 'f32', 'rw');
            const k = new Kernel(`
              fn main(gid: vec3u) {
                inout[gid.x] = inout[gid.x] * 10.0;
              }
            `, { threads: 4 });
            const node = v.pass(k, { inout: buf });
            v.run(node);
            const output = await v.read(node);
            return Array.from(output.inout);
        });

        // First 4 multiplied, last 4 untouched
        expect(result.slice(0, 4)).toEqual([10, 20, 30, 40]);
        expect(result.slice(4)).toEqual([100, 100, 100, 100]);
    });
});

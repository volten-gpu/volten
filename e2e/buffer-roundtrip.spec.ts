/**
 * E2E: Buffer Data Round-Trip Tests
 *
 * Verifies that data survives the full pipeline:
 * JS array → Buffer → GPU upload → compute (identity) → staging copy → readback → TypedArray
 *
 * These tests catch bugs in packing, upload, readback, and type conversion
 * that cannot be caught without a real GPUDevice.
 */
import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:5174';

test.describe('Buffer data round-trip', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForFunction(() => (window as any).__gpuReady === true, null, { timeout: 10_000 });
    });

    test('f32 data survives GPU round-trip', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, Kernel } = (window as any).Volten;
            const v = await volten();

            const data = [1.5, 2.25, 3.75, 4.0, 5.125];
            const buf = new Buffer(data, 'f32', 'rw');
            const k = new Kernel(`
              fn main(gid: vec3u) {
                inout[gid.x] = inout[gid.x];
              }
            `);
            const node = v.pass(k, { inout: buf });
            v.run(node);
            const output = await v.read(node);
            return Array.from(output.inout);
        });

        expect(result).toEqual([1.5, 2.25, 3.75, 4.0, 5.125]);
    });

    test('u32 data survives GPU round-trip', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, Kernel } = (window as any).Volten;
            const v = await volten();

            const data = [0, 1, 255, 65535, 4294967295]; // includes max u32
            const buf = new Buffer(data, 'u32', 'rw');
            const k = new Kernel(`
              fn main(gid: vec3u) {
                inout[gid.x] = inout[gid.x];
              }
            `);
            const node = v.pass(k, { inout: buf });
            v.run(node);
            const output = await v.read(node);
            return Array.from(output.inout);
        });

        expect(result).toEqual([0, 1, 255, 65535, 4294967295]);
    });

    test('i32 data with negative values survives GPU round-trip', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, Kernel } = (window as any).Volten;
            const v = await volten();

            const data = [-100, -1, 0, 1, 100];
            const buf = new Buffer(data, 'i32', 'rw');
            const k = new Kernel(`
              fn main(gid: vec3u) {
                inout[gid.x] = inout[gid.x];
              }
            `);
            const node = v.pass(k, { inout: buf });
            v.run(node);
            const output = await v.read(node);
            return Array.from(output.inout);
        });

        expect(result).toEqual([-100, -1, 0, 1, 100]);
    });

    test('larger buffer (1024 elements) round-trips correctly', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, Kernel } = (window as any).Volten;
            const v = await volten();

            // Create a predictable pattern
            const data = Array.from({ length: 1024 }, (_, i) => i * 0.5);
            const buf = new Buffer(data, 'f32', 'rw');
            const k = new Kernel(`
              fn main(gid: vec3u) {
                inout[gid.x] = inout[gid.x];
              }
            `);
            const node = v.pass(k, { inout: buf });
            v.run(node);
            const output = await v.read(node);
            const arr = Array.from(output.inout);

            // Verify first 5, last 5, and length
            return {
                length: arr.length,
                first5: arr.slice(0, 5),
                last5: arr.slice(-5),
            };
        });

        expect(result.length).toBe(1024);
        expect(result.first5).toEqual([0, 0.5, 1, 1.5, 2]);
        // last 5 elements: indices 1019..1023 → values 509.5, 510, 510.5, 511, 511.5
        expect(result.last5).toEqual([509.5, 510, 510.5, 511, 511.5]);
    });

    test('reading a Buffer directly (not through a Node) works', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, Kernel } = (window as any).Volten;
            const v = await volten();

            const buf = new Buffer([7, 14, 21], 'f32', 'rw');
            const k = new Kernel(`
              fn main(gid: vec3u) {
                inout[gid.x] = inout[gid.x] * 2.0;
              }
            `);
            const node = v.pass(k, { inout: buf });
            v.run(node);

            // Read the buffer directly instead of through the node
            const output = await v.read(buf);
            return Array.from(output);
        });

        expect(result).toEqual([14, 28, 42]);
    });
});

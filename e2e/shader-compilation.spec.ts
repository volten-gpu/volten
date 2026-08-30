/**
 * E2E: Shader Compilation Smoke Tests
 *
 * Verifies that Volten's assembled WGSL (binding declarations + kernel source +
 * workgroup injection + builtin shorthands) compiles successfully on a real GPU.
 * These catch malformed WGSL that vitest's mocks cannot detect.
 */
import { test, expect } from '@playwright/test';
const BASE_URL = 'http://localhost:5174';
test.describe('Shader compilation on real GPU', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForFunction(
            () => (window as any).__gpuReady === true,
            null,
            { timeout: 10000 }
        );
    });
    test('identity kernel compiles and runs', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, kernel } = (window as any).Volten;
            const v = await volten();
            const buf = new Buffer([1, 2, 3, 4], 'f32', 'rw');
            const identity = kernel({
                shader: `
              fn main(gid: vec3u) {
                inout[gid.x] = inout[gid.x];
              }
            `
            });
            const node = identity({ inout: buf });
            v.run(node);
            const output = await v.read(buf);
            return Array.from(output);
        });
        // Identity: output should equal input
        expect(result).toEqual([1, 2, 3, 4]);
    });
    test('kernel with all builtin shorthands (gid, lid, wid) compiles', async ({
        page
    }) => {
        // This test ensures the shorthand expansion produces valid WGSL
        const result = await page.evaluate(async () => {
            const { volten, Buffer, kernel } = (window as any).Volten;
            const v = await volten();
            const buf = new Buffer([0, 0, 0, 0], 'f32', 'rw');
            const writeBuiltinIds = kernel({
                shader: `
              fn main(gid: vec3u, lid: u32) {
                inout[gid.x] = f32(gid.x) + f32(lid);
              }
            `
            });
            const node = writeBuiltinIds({ inout: buf });
            v.run(node);
            const output = await v.read(buf);
            return Array.from(output);
        });
        // We don't assert exact values (lid depends on workgroup layout),
        // just that it didn't crash — the shader compiled and dispatched.
        expect(result).toHaveLength(4);
        expect(
            result.every((v: number) => typeof v === 'number' && !isNaN(v))
        ).toBe(true);
    });
    test('kernel with separate input/output buffers compiles', async ({
        page
    }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, kernel } = (window as any).Volten;
            const v = await volten();
            const input = new Buffer([10, 20, 30], 'f32');
            const output = new Buffer([0, 0, 0], 'f32', 'rw');
            const copyValues = kernel({
                shader: `
              fn main(gid: vec3u) {
                output[gid.x] = input[gid.x];
              }
            `,
                outputs: { output: { definedBy: 'input' } },
                threads: 'input'
            });
            const node = copyValues({ input, output });
            v.run(node);
            const res = await v.read(node);
            return Array.from(res.output);
        });
        expect(result).toEqual([10, 20, 30]);
    });
    test('kernel with u32 type compiles and runs', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, kernel } = (window as any).Volten;
            const v = await volten();
            const buf = new Buffer([5, 10, 15, 20], 'u32', 'rw');
            const identityU32 = kernel({
                shader: `
              fn main(gid: vec3u) {
                inout[gid.x] = inout[gid.x];
              }
            `
            });
            const node = identityU32({ inout: buf });
            v.run(node);
            const output = await v.read(buf);
            return Array.from(output);
        });
        expect(result).toEqual([5, 10, 15, 20]);
    });
    test('kernel with i32 type compiles and runs', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, kernel } = (window as any).Volten;
            const v = await volten();
            const buf = new Buffer([-1, 0, 1, 42], 'i32', 'rw');
            const identityI32 = kernel({
                shader: `
              fn main(gid: vec3u) {
                inout[gid.x] = inout[gid.x];
              }
            `
            });
            const node = identityI32({ inout: buf });
            v.run(node);
            const output = await v.read(buf);
            return Array.from(output);
        });
        expect(result).toEqual([-1, 0, 1, 42]);
    });
});

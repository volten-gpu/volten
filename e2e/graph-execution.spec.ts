/**
 * E2E: Graph Execution Tests
 *
 * Verifies that multi-node DAG execution works end-to-end on a real GPU:
 * - Linear chains (A → B)
 * - Multiple independent nodes v.run([A, B])
 * - Diamond/fork patterns
 * - Shared buffer synthetic dependency ordering
 *
 * These tests verify that node B actually reads the transformed output of node A,
 * something the mocked execution.test.ts cannot confirm.
 */
import { test, expect } from '@playwright/test';
const BASE_URL = 'http://localhost:5174';
test.describe('Linear chain execution', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForFunction(
            () => (window as any).__gpuReady === true,
            null,
            { timeout: 10000 }
        );
    });
    test('A → B: second node reads first node output', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, kernel } = (window as any).Volten;
            const v = await volten();
            // A: multiply by 2  (input: [1,2,3,4] → mid: [2,4,6,8])
            const input = new Buffer([1, 2, 3, 4], 'f32');
            const mid = new Buffer([0, 0, 0, 0], 'f32', 'rw');
            const doubleValues = kernel({
                shader: `
              fn main(gid: vec3u) {
                mid[gid.x] = input[gid.x] * 2.0;
              }
            `,
                outputs: { mid: { definedBy: 'input' } },
                threads: 'input'
            });
            // B: add 100  (mid: [2,4,6,8] → output: [102,104,106,108])
            const output = new Buffer([0, 0, 0, 0], 'f32', 'rw');
            const addHundred = kernel({
                shader: `
              fn main(gid: vec3u) {
                output[gid.x] = data[gid.x] + 100.0;
              }
            `,
                outputs: { output: { definedBy: 'data' } },
                threads: 'data'
            });
            const A = doubleValues({ input, mid });
            const B = addHundred({ data: A.mid, output });
            v.run(B);
            const res = await v.read(B);
            return Array.from(res.output);
        });
        // 1*2+100=102, 2*2+100=104, 3*2+100=106, 4*2+100=108
        expect(result).toEqual([102, 104, 106, 108]);
    });
    test('A → B → C: three-node chain', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, kernel } = (window as any).Volten;
            const v = await volten();
            const buf1 = new Buffer([1, 2, 3, 4], 'f32');
            const buf2 = new Buffer([0, 0, 0, 0], 'f32', 'rw');
            const buf3 = new Buffer([0, 0, 0, 0], 'f32', 'rw');
            const buf4 = new Buffer([0, 0, 0, 0], 'f32', 'rw');
            // A: +10
            const addTen = kernel({
                shader: `
              fn main(gid: vec3u) { out[gid.x] = src[gid.x] + 10.0; }
            `,
                outputs: { out: { definedBy: 'src' } },
                threads: 'src'
            });
            // B: *3
            const tripleValues = kernel({
                shader: `
              fn main(gid: vec3u) { out[gid.x] = src[gid.x] * 3.0; }
            `,
                outputs: { out: { definedBy: 'src' } },
                threads: 'src'
            });
            // C: -1
            const subtractOne = kernel({
                shader: `
              fn main(gid: vec3u) { out[gid.x] = src[gid.x] - 1.0; }
            `,
                outputs: { out: { definedBy: 'src' } },
                threads: 'src'
            });
            const A = addTen({ src: buf1, out: buf2 });
            const B = tripleValues({ src: A.out, out: buf3 });
            const C = subtractOne({ src: B.out, out: buf4 });
            v.run(C);
            const res = await v.read(C);
            return Array.from(res.out);
        });
        // (1+10)*3-1=32, (2+10)*3-1=35, (3+10)*3-1=38, (4+10)*3-1=41
        expect(result).toEqual([32, 35, 38, 41]);
    });
});

test.describe('Plan execution', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForFunction(
            () => (window as any).__gpuReady === true,
            null,
            { timeout: 10000 }
        );
    });

    test('a callable plan lowers into multiple GPU dispatches', async ({
        page
    }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, kernel, plan } = (window as any).Volten;
            const v = await volten();
            const input = new Buffer([1, 2, 3, 4], 'f32');
            const mid = new Buffer([0, 0, 0, 0], 'f32', 'rw');
            const output = new Buffer([0, 0, 0, 0], 'f32', 'rw');

            const double = kernel({
                shader: `fn main(gid: vec3u) {
                    output[gid.x] = input[gid.x] * 2.0;
                }`,
                threads: 'input'
            });
            const addTen = kernel({
                shader: `fn main(gid: vec3u) {
                    output[gid.x] = input[gid.x] + 10.0;
                }`,
                threads: 'input'
            });
            const doubleThenAdd = plan((_, inputs) => {
                const A = double({ input: inputs.input, output: mid });
                const B = addTen({ input: A.output, output });
                return { result: B.output };
            });

            const A = doubleThenAdd({ input });
            v.run(A);
            const readback = await v.read(A);
            return Array.from(readback.result);
        });

        expect(result).toEqual([12, 14, 16, 18]);
    });
});
test.describe('Multi-node independent execution', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForFunction(
            () => (window as any).__gpuReady === true,
            null,
            { timeout: 10000 }
        );
    });
    test('v.run([A, B]) executes two independent nodes', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, kernel } = (window as any).Volten;
            const v = await volten();
            const buf1 = new Buffer([1, 2, 3], 'f32', 'rw');
            const buf2 = new Buffer([10, 20, 30], 'f32', 'rw');
            const doubleValues = kernel({
                shader: `
              fn main(gid: vec3u) { inout[gid.x] = inout[gid.x] * 2.0; }
            `
            });
            const addFive = kernel({
                shader: `
              fn main(gid: vec3u) { inout[gid.x] = inout[gid.x] + 5.0; }
            `
            });
            const A = doubleValues({ inout: buf1 });
            const B = addFive({ inout: buf2 });
            v.run([A, B]);
            const [resA, resB] = await v.read([buf1, buf2]);
            return {
                a: Array.from(resA),
                b: Array.from(resB)
            };
        });
        expect(result.a).toEqual([2, 4, 6]);
        expect(result.b).toEqual([15, 25, 35]);
    });
});
test.describe('Shared buffer execution ordering', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForFunction(
            () => (window as any).__gpuReady === true,
            null,
            { timeout: 10000 }
        );
    });
    test('v.run([A, B]) with shared rw buffer: A runs before B', async ({
        page
    }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, kernel } = (window as any).Volten;
            const v = await volten();
            // Both nodes write to the same buffer.
            // v.run([A, B]) means A should execute first (positional priority).
            // A writes gid.x * 10, B writes gid.x * 100.
            // If ordering is correct: A writes first, then B overwrites → final = gid.x * 100
            const shared = new Buffer([0, 0, 0, 0], 'f32', 'rw');
            const writeTens = kernel({
                shader: `
              fn main(gid: vec3u) { inout[gid.x] = f32(gid.x) * 10.0; }
            `
            });
            const writeHundreds = kernel({
                shader: `
              fn main(gid: vec3u) { inout[gid.x] = f32(gid.x) * 100.0; }
            `
            });
            const A = writeTens({ inout: shared });
            const B = writeHundreds({ inout: shared });
            // A before B in positional order
            v.run([A, B]);
            const output = await v.read(shared);
            return Array.from(output);
        });
        // B runs after A, so B's values (gid * 100) should be the final result
        expect(result).toEqual([0, 100, 200, 300]);
    });
});
test.describe('v.wait() completion', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForFunction(
            () => (window as any).__gpuReady === true,
            null,
            { timeout: 10000 }
        );
    });
    test('v.wait() resolves after GPU work completes', async ({ page }) => {
        const result = await page.evaluate(async () => {
            const { volten, Buffer, kernel } = (window as any).Volten;
            const v = await volten();
            const buf = new Buffer([1, 2, 3], 'f32', 'rw');
            const multiplyByFive = kernel({
                shader: `
              fn main(gid: vec3u) { inout[gid.x] = inout[gid.x] * 5.0; }
            `
            });
            const node = multiplyByFive({ inout: buf });
            // wait() should fully complete before we read
            await v.wait(node);
            const output = await v.read(buf);
            return Array.from(output);
        });
        expect(result).toEqual([5, 10, 15]);
    });
});

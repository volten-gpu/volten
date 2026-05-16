import { volten, Buffer, Kernel, Uniform } from '@volten/core';

async function main(): Promise<void> {
    const v = await volten();
    const inout = new Buffer([1, 2, 3, 4], 'f32', 'rw');
    const mult = new Uniform(10, 'f32');

    const kernel = new Kernel(`
      fn main(gid: vec3u) {
        inout[gid.x] = inout[gid.x] * mult;
      }
    `);

    const A = v.pass(kernel, { inout, mult });
    v.run(A);

    console.log(await v.read(inout));
}

main();

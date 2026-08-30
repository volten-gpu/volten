import { volten, Buffer, kernel, Uniform } from '@volten/core';
async function main(): Promise<void> {
    const v = await volten();
    const inout = new Buffer([1, 2, 3, 4], 'f32', 'rw');
    const mult = new Uniform(10, 'f32');
    const scaleValues = kernel({
        shader: `
      fn main(gid: vec3u) {
        inout[gid.x] = inout[gid.x] * mult;
      }
    `
    });
    const A = scaleValues({ inout, mult });
    v.run(A);
    console.log(await v.read(inout));
}
main();

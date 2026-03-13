import { struct, volten, Buffer, Kernel, Uniform } from '@volten/core';

const v = await volten();

// Buffer has 8 elements, but we only process the first 4
const buf = new Buffer([1, 2, 3, 4, 100, 100, 100, 100], 'f32', 'rw');
const mult = new Uniform(10, 'f32');

const k = new Kernel(
    `
        fn main(gid: vec3u) {
          inout[gid.x] = inout[gid.x] * mult;
        }
      `,
    { threads: 4 }
);
const node = v.pass(k, { inout: buf, mult });

v.run(node);

const output = await v.read(node);

console.log(Array.from(output.inout));

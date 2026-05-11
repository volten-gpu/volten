import { volten, Buffer, Kernel, Uniform } from '@volten/core';

const v = await volten();

// Buffer has 8 elements, but we only process the first 4
const buf = new Buffer([1, 2, 3, 4, 100, 100, 100, 100], 'f32', 'rw');
const mult = new Uniform(10, 'f32');

const k = new Kernel(
    `
        fn main(gid: vec3u) {
          if (gid.x == 2) {
            enableDebug();
            debugF32("test message", inout[gid.x] * mult);
          }       
               
          if (gid.x == 1) {
            enableDebug();
            debugVec4f("test message 2", vec4f(0.0, 1.0, 2.0, 3.0));
          }

          inout[gid.x] = inout[gid.x] * mult;
        }
      `,
    { threads: 4, label: 'Ktest' }
);
const A = v.pass(k, { inout: buf, mult }, { debug: {} });
// const B = v.pass(k, { inout: buf, mult });

// v.run([A, B]);
v.run(A);

const output = await v.read(buf);
console.log(Array.from(output));

const debugRes = await v.readDebug(A);
debugRes.print();

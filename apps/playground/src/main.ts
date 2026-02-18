import { volten, Buffer, Kernel } from '@volten/core';

let v = await volten();

let k = new Kernel(`
  fn main(gid: vec3u) {
    inout[gid.x] = inout[gid.x] * 2.0;
  }`,
  {
    threads: "inout"
  }
);

let buffer1 = new Buffer([1, 2, 3, 4], "f32", "rw");
let buffer2 = new Buffer([1, 2, 3, 4], "f32", "rw");

let A = v.pass(k, { inout: buffer1 });
let B = v.pass(k, { inout: buffer2 });
let C = v.pass(k, { inout: A.inout });

v.run(A, B, C);
let result = await v.read(A);
console.log(result);

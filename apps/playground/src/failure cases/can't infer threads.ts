import { volten, Buffer, Kernel } from '@volten/core';

let v = await volten();

let k = new Kernel(`
  fn main(gid: vec3u) {
    inout[gid.x] = inout[gid.x] * 2.0;
  }`
);

let buffer = new Buffer([1, 2, 3, 4], "f32", "rw");

let A = v.pass(k, { inout: buffer });
let B = v.pass(k, { inout: buffer });
let C = v.pass(k, { inout: A.inout });

v.run(A, B, C);
let result = await v.read(C);

console.log(result);

/*

Uncaught Error: Volten Error: Cannot auto-infer thread count — no buffer bindings found.

*/
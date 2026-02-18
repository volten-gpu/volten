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

let buffer = new Buffer([1, 2, 3, 4], "f32", "rw");

let A = v.pass(k, { inout: buffer });
let B = v.pass(k, { inout: buffer });
let C = v.pass(k, { inout: A.inout });

v.run(A, B, C);
let result = await v.read(C);

console.log(result);

/*

context.ts:348 Uncaught Error: Volten Error: Output "inout" is not a Buffer/RawBuffer — cannot read back.
    at VoltenContext.read (context.ts:348:23)
    at main.ts:21:22

*/
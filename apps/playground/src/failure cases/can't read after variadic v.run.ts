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

v.run(
    v.pass(k, { inout: buffer }),
    v.pass(k, { inout: buffer }),
    v.pass(k, { inout: buffer })
);

// ??? how do I read?
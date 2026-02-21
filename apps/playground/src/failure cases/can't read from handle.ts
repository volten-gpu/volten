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

v.run([A, B, C]);
let result = await v.read(C);

console.log(result);

/*

context.ts:348 Uncaught Error: Volten Error: Output "inout" is not a Buffer/RawBuffer — cannot read back.
    at VoltenContext.read (context.ts:348:23)
    at main.ts:21:22

*/

// that's a good point ^ but what do we return when we pass multiple
// values?

/* 
I think we should move to a slightly different, more consistent system:

Right now, I support the unpacked syntax on v.run:
v.run(a, b, c)

which doesn't require having to think about return types, but on the other hand:
v.read(a, b, c) 

would either always return an array (which doesn't look great when passing a single value) 
or would return one object if only one value is passed, but that seems like something that could 
be unexpected for the user. Ideally it would either be non array or array syntax:

let res   = v.read(buffer);     // we know it will be only one object
let array = v.read([buffer]);   // we know it returns an array because we passed an array
let array = v.read([buffer, nodeA, nodeB]); // same ^

and only allow the array syntax when we need multiple values, and that will always return
an array even if a single value is provided, but that, to be sort of like coherent 
with the other methods, would require me to change:
v.run to follow the same standard.

either v.run(A)
or     v.run([A, ...])

the additional benefit is that if we ever need to pass actual options to v.run, the
second argument would be able to be an actual options object instead of another node

*/
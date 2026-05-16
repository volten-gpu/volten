# @volten/core

WebGPU compute, without the ceremony.

`@volten/core` is the core Volten package: a TypeScript API for writing WGSL compute kernels, binding buffers and uniforms, scheduling passes, reading results back to the CPU, shader debugging and more.

## Installation

pnpm add @volten/core

## Quick Start

```javascript
import { volten, Buffer, Kernel, Uniform } from '@volten/core';

const v = await volten();
const inout = new Buffer([1, 2, 3, 4], 'f32', 'rw');
const mult = new Uniform(10, 'f32');

const kernel = new Kernel(`
  fn main(gid: vec3u) {
    inout[gid.x] = inout[gid.x] * mult;
  }
`);

const node = v.pass(kernel, { inout, mult });
v.run(node);

console.log(await v.read(inout));
// Float32Array [10, 20, 30, 40]
```

## Documentation

Full documentation is in progress. See the main repository README:
https://github.com/volten-gpu/volten

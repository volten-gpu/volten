<p align="center">
  <img
    src="./docs/assets/logo.png"
    width="180"
    alt="Volten logo"
  />
</p>

# Volten
> WebGPU made easy.

Volten simplifies WebGPU compute shader development with an API designed for logic, not logistics.

### Overview

This is how simple it is in Volten to run a compute shader and read back the result:

```javascript
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

console.log( await v.read(inout) );
```

But there's more, Volten has many utilities to speed up compute shader development, including first-class support for shader debugging

```javascript
const kernel = new Kernel(`
  fn main(gid: vec3u) {
    inout[gid.x] = inout[gid.x] * mult;

    if (gid.x == 2) {
      enableDebug();
      debugF32("f32 value debug", inout[gid.x]);
    }  
  }
`);

const node = v.pass(kernel, { inout, mult }, { debug: true });
v.run(node);

const debugRes = await v.readDebug(node);
debugRes.print();
// prints: [2,0,0] f32 value debug: 30
```

### Installation

### Documentation

### Packages

### License

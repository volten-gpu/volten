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

### Quick start

This is how simple it is in Volten to run a compute shader and read back the result:

```ts
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

<br>

But there's more. Volten includes utilities that speed up compute shader development, including first-class support for shader debugging:

```ts
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

Install the public core package:

```sh
pnpm add @volten/core
```

Or with another package manager:

```sh
npm install @volten/core
yarn add @volten/core
bun add @volten/core
```

### Documentation

Documentation is still in progress: [docs WIP](./docs/README.md).

### Packages

| Package          | Status                    | Description                                                                                                            |
| ---------------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `@volten/core`   | Public                    | Core WebGPU compute API: contexts, kernels, buffers, uniforms, pass graphs, scheduling, readback, and debug utilities. |
| `@volten/stdlib` | Private workspace package | Reserved for reusable compute kernels and higher-level GPU operations. Not part of the first public release yet.       |
| `@volten/math`   | Private workspace package | Reserved for future math helpers. Not part of the first public release yet.                                            |

### Development

This repository uses pnpm workspaces and Turborepo.

```sh
pnpm install
pnpm build
pnpm lint
pnpm check-types
pnpm test:all
```

Useful scripts:

| Command            | Description                                                     |
| ------------------ | --------------------------------------------------------------- |
| `pnpm build`       | Build workspace packages.                                       |
| `pnpm lint`        | Run lint checks.                                                |
| `pnpm check-types` | Run TypeScript checks.                                          |
| `pnpm test:e2e`    | Build `@volten/core` and run browser E2E tests with Playwright. |
| `pnpm test:all`    | Run build, package tests, and E2E tests.                        |
| `pnpm test:pack`   | Smoke-test package packing.                                     |

### License

MIT. See [LICENSE](./LICENSE).

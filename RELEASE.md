# Release Checklist

Run these commands from the repository root before shipping a Volten release:

```sh
pnpm build
pnpm lint
pnpm check-types
pnpm test:all
pnpm test:pack
cd packages/core
npm pack --dry-run
```

`@volten/core` declares `"sideEffects": false` for bundlers. Keep package
imports inert: source files may define and export values at module load time,
but WebGPU work, environment access, global mutation, logging, and async startup
must stay behind explicit user calls such as `volten()`, `v.pass()`, or
`v.run()`.

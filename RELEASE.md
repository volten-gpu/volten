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
must stay behind explicit user calls such as `volten()`, operation invocation,
or `v.run()`.

Additional checklist:

- Ensure `main` is green.
- Decide the next `@volten/core` version.
- Update `packages/core/package.json`.
- Update `CHANGELOG.md`.
- Commit the version and changelog changes.

## Create Tag

```sh
git tag v0.1.0
git push origin v0.1.0
```

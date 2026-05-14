# Release Checklist

Run these commands from the repository root before shipping a Volten release:

```sh
pnpm build
pnpm lint
pnpm check-types
pnpm test:all
cd packages/core
npm pack --dry-run
```

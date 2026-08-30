# Changelog

## [0.2.0] - 2026-08-31

### Added

- Reusable `plan()` operations for composing multi-kernel compute graphs.
- Context-dependent shader selectors with access to the active GPU device,
  features, and limits.

### Changed

- **Breaking:** Replaced `new Kernel()` and `v.pass()` with callable,
  context-free `kernel()` operations that materialize lazily.
- Refactored graph compilation and scheduling around handles, including
  automatic upstream scheduling and safe ordering for shared buffers.
- Updated readback, debugging, examples, documentation, and test coverage for
  the new operation model.

## [0.1.0] - 2026-05-26

### Added

- Initial public release of `@volten/core`.

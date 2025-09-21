# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0]

### Added
- Divergence types are now callable, enabling `div(x, y)` and `div(x)` syntax
- Broadcasting support for divergences using `div.(x, y)`
- Backward compatibility for the old `evaluate(div, x, y)` API with deprecation warnings

### Changed
- Divergences now inherit from `PreMetric` (via `Distances.jl`) instead of being standalone types
- Improved performance through streamlined divergence evaluation
- Updated constructor signatures for modified divergences

### Deprecated
- `evaluate(div, x, y)` and `evaluate(div, x)` functions are deprecated in favor of callable syntax `div(x, y)` and `div(x)`

### Fixed
- Type annotation issues in gradient and hessian functions
- Improved numerical stability in divergence calculations

## [0.3.0] - Previous Release
- Initial stable release with basic divergence functionality

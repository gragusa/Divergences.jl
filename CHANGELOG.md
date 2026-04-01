# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.3] (Unreleased)

Bug Fixes

  - Type-promotion in reduction accumulators — fixed accumulator type-promotion so integer arrays work correctly in dual reductions; avoided allocation in dual_from_primal
  - Hessian formula — fixed hessian(d, a, b) to properly divide by bᵢ
  - In-place method type inconsistencies — fixed type mismatches in in-place methods
  - Type stability improvements — improved type stability across the package
  - CressieRead type description typo — corrected docstring

Refactoring & Performance

  - Collapsed @eval-generated call overloads into generic AbstractDivergence methods, simplifying the dispatch logic
  - Improved χ² gradient performance
  - Removed ForwardDiff and Test from direct dependencies in Project.toml (moved to test/extensions)

Documentation

  - Replaced old doc/ directory with a proper Documenter.jl setup under docs/ including:
    - API reference (api.md)
    - Theory page (theory.md)
    - Computation details (computation.md)
    - Divergence catalog (divergences.md)

Infrastructure & CI

  - New CI workflow — modernized ci.yml, added CompatHelper.yml, TagBot.yml
  - Benchmarking — added benchmark/ suite with BenchmarkTools and an ASV runner (benchmark.yml workflow)
  - SciML code style — adopted JuliaFormatter with SciML style, added badge to README
  - Pre-commit hook config added
  - Expanded .gitignore

Tests

  - Major test expansion — new test/runtests.jl (+566 lines) and test/test_duals.jl (+169 lines) covering dual-number correctness and accumulator type-promotion with integer arrays


## [0.4.2]

- Implement Legendre-Fenchel conjugates ψ(v) for all divergence types, enabling dual formulations for optimization problems.
  
## [0.4.1]

- Fix several critical issues introduced by mistakes in the previous version.

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

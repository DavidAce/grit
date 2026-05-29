# AGENTS.md

GRIT is a C++23 eigensolver library extracted from xDMRGpp. It keeps reusable standard and generalized eigensolver logic in GRIT while leaving xDMRGpp-specific DMRG logic in user code.

The xDMRGpp source is at `https://github.com/DavidAce/xDMRGpp`.

## Project Layout

- `include/grit/`: public headers and template implementation files.
- `include/grit/internal/`: algorithms, form implementations, iterative solvers, and Jacobi-Davidson support.
- `source/`: explicit template instantiation support.
- `tests/`: Catch2 tests.
- `bench/standard/`: standard eigenvalue benchmark using Matrix Market input and optional HDF5 save/load.
- `cmake/`: dependency providers and package-manager support.

## Local Skills

Read the relevant project skill before making focused changes:

- `.skills/grit-build-test/SKILL.md`: CMake presets, package managers, and validation commands.
- `.skills/grit-xdmrgpp-port/SKILL.md`: rules for porting solver code from xDMRGpp.
- `.skills/grit-benchmark/SKILL.md`: benchmark structure and implementation preferences.
- `.skills/grit-commit-style/SKILL.md`: commit message style for this repository.

## Notation

- `l2`: Euclidean inner product, e.g. `V.adjoint() * V = I`.
- `bm`: B-metric inner product for generalized problems, e.g. `V.adjoint() * B * V = I`.
- Do not use the old xDMRGpp name `h2` in GRIT APIs or options.

## Defaults

- Preserve xDMRGpp solver logic closely when porting it.
- Keep DMRG-specific logic out of GRIT.
- Use existing CMake presets. Do not create ad hoc build directories.
- Use `std::print` and `std::format`; do not add `<iostream>` or `<iomanip>`.
- Keep changes scoped and add tests when solver behavior changes.
- Prefer `snake_case` for functions, variables, fields, files, and user-facing options.
- Reserve `CamelCase` for types and enum names.

---
name: grit-benchmark
description: Use when changing GRIT benchmarks, especially bench/standard, benchmark dependencies, CLI options, matrix loading, or HDF5 save/load.
---

# Benchmarks

Benchmark source should be split by responsibility. Avoid growing one large `main.cpp`.

- `bench/standard/` benchmarks standard eigenvalue problems.
- Matrix input is a file path, not a named matrix shortcut.
- Matrix Market is acceptable for SuiteSparse-style matrices because it is simple to read without heavy dependencies.
- HDF5 is acceptable for saving and loading initial guesses and eigenvectors.
- Local benchmark data belongs under `bench/data/` and should not be committed.
- CLI parsing should use CLI11.
- Use `std::print` and `std::format`.
- Do not add `<iostream>` or `<iomanip>`.
- Benchmark defaults should not stop on eigenvalue or residual stalls unless the user sets stall tolerances.
- Prefer concise table-style output for benchmark repetitions.

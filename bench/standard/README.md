# Standard Eigenvalue Benchmark

This benchmark reads a local Matrix Market file, builds a sparse matrix-vector
callback, and runs `grit::standard::gdplusk` for the smallest algebraic
eigenpair.

## Example Matrix

The default matrix path points to `finance256` from the SuiteSparse Matrix Collection.
Benchmark data is intentionally not tracked by git.

```bash
mkdir -p bench/data
curl -L \
  https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/finance256.tar.gz \
  -o bench/data/finance256.tar.gz
tar -xzf bench/data/finance256.tar.gz -C bench/data
rm bench/data/finance256.tar.gz
```

## Build

```bash
cmake --preset kraken -DGRIT_ENABLE_BENCH=ON
cmake --build --preset kraken --target grit-bench-standard
```

## Run

```bash
./build/Release/bench/standard/grit-bench-standard \
  --matrix bench/data/finance256/finance256.mtx
```

Use any full or relative path to a compatible Matrix Market file.

## Warm Start

Save a partial cheap-Olsen solution:

```bash
./build/Release/bench/standard/grit-bench-standard \
  --matrix bench/data/finance256/finance256.mtx \
  --residual-correction=cheap-olsen \
  --tol=1e-3 \
  --save-eigvec=bench/data/finance256/warmstart.h5
```

Use it as the initial guess in a later run:

```bash
./build/Release/bench/standard/grit-bench-standard \
  --matrix bench/data/finance256/finance256.mtx \
  --initial-guess=bench/data/finance256/warmstart.h5 \
  --residual-correction=jacobi-davidson \
  --tol=1e-5 \
  --refined-rayleigh-ritz
```

The HDF5 dataset path is `/grit/standard/eigvecs`.

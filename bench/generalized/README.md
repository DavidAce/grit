# Generalized Eigenvalue Benchmark

This benchmark reads a Matrix Market pair and runs one of:

- `grit::generalized::gdplusk`
- `grit::generalized::lanczos`
- `grit::generalized::lobpcg`

for the generalized problem `A x = lambda B x`.

`A` must be symmetric or Hermitian. `B` must be symmetric or Hermitian positive
definite.

## Matrix Pairs

Benchmark data is intentionally not tracked by git. Store downloaded matrices
under `bench/data/`.

Good starter pairs are structural stiffness/mass problems from the Matrix
Market Harwell-Boeing collection. For example, `BCSSTK19` is a real symmetric
stiffness matrix and `BCSSTM19` is the corresponding real symmetric positive
definite mass matrix.

Use the stiffness matrix as `--matrix-a-path` and the mass matrix as
`--matrix-b-path`.

## Build

```bash
cmake --preset kraken -DGRIT_ENABLE_BENCH=ON
cmake --build --preset kraken --target grit-bench-generalized
```

## Run

```bash
./build/Release/bench/generalized/grit-bench-generalized \
  --algo gdplusk \
  --matrix-a-path bench/data/bcsstk19/bcsstk19.mtx \
  --matrix-b-path bench/data/bcsstm19/bcsstm19.mtx
```

Use `--algo` to select the solver:

- `gdplusk`
- `lanczos`
- `lobpcg`

`gdplusk` is the default.

## Generalized Options

Use `--use-b-inner-product` to use B-metric inner products. It can also be swept:

```bash
--use-b-inner-product=[false,true]
```

Use `--use-jd-b-only` with `--algo=gdplusk` to benchmark the B-only
Jacobi-Davidson correction path:

```bash
./build/Release/bench/generalized/grit-bench-generalized \
  --algo gdplusk \
  --matrix-a-path bench/data/bcsstk19/bcsstk19.mtx \
  --matrix-b-path bench/data/bcsstm19/bcsstm19.mtx \
  --residual-correction=jacobi-davidson \
  --use-b-inner-product \
  --use-jd-b-only
```

`--use-jd-b-only` is rejected for `lanczos` and `lobpcg`.

## Sweeps

The generalized benchmark follows the standard benchmark CLI style. Common sweep
axes include:

```bash
--ncv=[8,16,32]
--block-size=[1,2]
--ritz=[SR,LM]
--tol=[1e-8,1e-10]
--use-b-inner-product=[false,true]
```

GD+K also supports sweeps over residual correction and inner-solver controls:

```bash
--residual-correction=[cheap-olsen,jacobi-davidson,auto]
--inner-tol=[1e-1,1e-3]
--inner-max-iters=[100,1000]
```

## Warm Start

Save eigenvectors:

```bash
./build/Release/bench/generalized/grit-bench-generalized \
  --algo gdplusk \
  --matrix-a-path bench/data/bcsstk19/bcsstk19.mtx \
  --matrix-b-path bench/data/bcsstm19/bcsstm19.mtx \
  --residual-correction=cheap-olsen \
  --tol=1e-3 \
  --save-eigvec=bench/data/bcs19/warmstart.h5
```

Use them as an initial guess:

```bash
./build/Release/bench/generalized/grit-bench-generalized \
  --algo gdplusk \
  --matrix-a-path bench/data/bcsstk19/bcsstk19.mtx \
  --matrix-b-path bench/data/bcsstm19/bcsstm19.mtx \
  --initial-guess=bench/data/bcs19/warmstart.h5 \
  --residual-correction=jacobi-davidson \
  --tol=1e-5
```

Saved eigvecs are stored under `/grit/generalized/<algo>/eigvecs`.

## Results

Use `--save-results=<file>` to store final result rows and solver snapshots.
Use `--print-summary --save-results=<file>` to print a summary without rerunning
the benchmark.

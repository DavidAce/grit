# Standard Eigenvalue Benchmark

This benchmark reads a local Matrix Market file, builds a sparse matrix-vector
callback, and runs `grit::standard::gdplusk` for the smallest algebraic
eigenpair.

## Example Matrix

The default matrix path points to `finance256` from the SuiteSparse Matrix Collection.
Benchmark data is intentionally not tracked by git.

```bash
mkdir -p bench/data
curl -L https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/finance256.tar.gz -o bench/data/finance256.tar.gz
tar -xzf bench/data/finance256.tar.gz -C bench/data
rm bench/data/finance256.tar.gz
```

## Additional SuiteSparse Matrices

Several matrices used in PRIMME benchmark papers are available from the
SuiteSparse Matrix Collection in Matrix Market format. Download them directly
under `bench/data/` and pass the extracted `.mtx` file with `--matrix-path`.

`finan512`: 74752 x 74752, 596992 nonzeros.

```bash
mkdir -p bench/data
curl -L  https://suitesparse-collection-website.herokuapp.com/MM/Mulvey/finan512.tar.gz -o bench/data/finan512.tar.gz
tar -xzf bench/data/finan512.tar.gz -C bench/data
rm bench/data/finan512.tar.gz
```

Run with:

```bash
./build/Release/bench/standard/grit-bench-standard \
  --matrix-path bench/data/finan512/finan512.mtx
```

`Andrews`: 60000 x 60000, 760154 nonzeros.

```bash
mkdir -p bench/data
curl -L https://suitesparse-collection-website.herokuapp.com/MM/Andrews/Andrews.tar.gz -o bench/data/Andrews.tar.gz
tar -xzf bench/data/Andrews.tar.gz -C bench/data
rm bench/data/Andrews.tar.gz
```

Run with:

```bash
./build/Release/bench/standard/grit-bench-standard \
  --matrix-path bench/data/Andrews/Andrews.mtx
```

`cfd2`: 123440 x 123440, 3085406 nonzeros.

```bash
mkdir -p bench/data
curl -L https://suitesparse-collection-website.herokuapp.com/MM/Rothberg/cfd2.tar.gz -o bench/data/cfd2.tar.gz
tar -xzf bench/data/cfd2.tar.gz -C bench/data
rm bench/data/cfd2.tar.gz
```

Run with:

```bash
./build/Release/bench/standard/grit-bench-standard \
  --matrix-path bench/data/cfd2/cfd2.mtx
```

## Build

```bash
cmake --preset kraken -DGRIT_ENABLE_BENCH=ON
cmake --build --preset kraken --target grit-bench-standard
```

## Run

```bash
./build/Release/bench/standard/grit-bench-standard \
  --matrix-path bench/data/finance256/finance256.mtx
```

Use any full or relative path to a compatible Matrix Market file.

## AUTO Residual Correction

`--residual-correction=auto` starts with cheap Olsen. Once cheap Olsen has
dwelled for `--auto-min-dwell-iters` steps and both the eigenvalue and residual
histories saturate, AUTO switches to Jacobi-Davidson. AUTO also switches to
Jacobi-Davidson once the residual norm is below
`--auto-jd-start-rnorm-threshold`; set that option to `0` to disable this
absolute residual trigger.

While Jacobi-Davidson is active, AUTO periodically forces one cheap-Olsen probe.
If that probe improves the selected Ritz value by more than
`--auto-cheap-probe-factor * max(rnorm^2, roundoff scale)`, cheap Olsen
continues. Otherwise AUTO returns to Jacobi-Davidson.

The main AUTO controls are `--auto-sat-eigval-threshold`,
`--auto-sat-rnorm-threshold`, `--auto-jd-start-rnorm-threshold`,
`--auto-min-dwell-iters`, `--auto-cheap-probe-interval`, and
`--auto-cheap-probe-factor`.

`--auto-sat-eigval-threshold` is relative to the average absolute eigenvalue
magnitude in the recent history window, so `1e-3` means roughly that the recent
Ritz values agree to about three significant digits.

## Warm Start

Save a partial cheap-Olsen solution:

```bash
./build/Release/bench/standard/grit-bench-standard \
  --matrix-path bench/data/finance256/finance256.mtx \
  --residual-correction=cheap-olsen \
  --tol=1e-3 \
  --save-eigvec=bench/data/finance256/warmstart.h5
```

Use it as the initial guess in a later run:

```bash
./build/Release/bench/standard/grit-bench-standard \
  --matrix-path bench/data/finance256/finance256.mtx \
  --initial-guess=bench/data/finance256/warmstart.h5 \
  --residual-correction=jacobi-davidson \
  --tol=1e-5 \
  --refined-rayleigh-ritz
```

The HDF5 dataset path is `/grit/standard/eigvecs`.

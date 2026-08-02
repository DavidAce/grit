# Standard Eigenvalue Benchmark

This benchmark reads a local Matrix Market file, builds a sparse matrix-vector callback, and runs one of:

- `grit::standard::gdplusk`
- `grit::standard::lanczos`
- `grit::standard::lobpcg`

for the smallest algebraic eigenpair.

## Example Matrix

The default matrix path points to `finance256` from the SuiteSparse Matrix Collection. Benchmark data is intentionally
not tracked by git.

```bash
mkdir -p bench/data
curl -L https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/finance256.tar.gz -o bench/data/finance256.tar.gz
tar -xzf bench/data/finance256.tar.gz -C bench/data
rm bench/data/finance256.tar.gz
```

## Additional SuiteSparse Matrices

Several matrices used in PRIMME benchmark papers are available from the SuiteSparse Matrix Collection in Matrix Market
format. Download them directly under `bench/data/` and pass the extracted `.mtx` file with `--matrix-path`.

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
  --algo gdplusk \
  --matrix-path bench/data/finance256/finance256.mtx
```

Use any full or relative path to a compatible Matrix Market file.

## Algorithm Selection

Use `--algo` to select the solver:

```bash
./build/Release/bench/standard/grit-bench-standard \
  --algo lanczos \
  --matrix-path bench/data/finance256/finance256.mtx
```

Supported values are:

- `gdplusk`
- `lanczos`
- `lobpcg`

`gdplusk` is the default.

The following options are `gdplusk`-only and will be rejected for `lanczos` and
`lobpcg`:

- `--residual-correction`
- `--inner-max-iters`
- `--inner-tol`
- `--use-adaptive-inner-tolerance`
- all `--auto-*` options

With `--reltol` enabled, GRIT records a separate residual reference for each requested Ritz pair after that Ritz value
stabilizes. The test starts after three entries and uses the available rolling history, which retains up to five entries.
The relative target is `reltol * reference residual`; before a reference exists, only `--abstol` applies.
`--ritz-stabilization-tolerance` controls the common stabilization test for every solver.

## AUTO Residual Correction

`--residual-correction=auto` starts with cheap Olsen and switches to Jacobi-Davidson when every unconverged Ritz value is
stabilized. The same rolling-history test starts after three entries and is evaluated after every outer iteration. For Ritz
pair `i`, AUTO compares

```text
stddev(lambda_i) * norm(B * v_i) / norm(r_i)
```

with `--ritz-stabilization-tolerance`, using `B * v_i = v_i` for standard problems. The numerator is the residual perturbation
caused by varying only the Ritz value, while the denominator is the current residual uncertainty. This ratio is unchanged
by equivalent scaling of a standard or generalized problem. Residual oscillation therefore does not prevent JD activation
once the Ritz values have stabilized relative to the residual.

While JD is active, AUTO performs a probe of `--auto-probe-length` consecutive cheap Olsen outer iterations every
`--auto-probe-interval` JD outer iterations. The rolling stabilization test continues during the probe. After the requested
number of probe iterations, AUTO resumes JD if every unconverged Ritz value is stabilized; otherwise it continues with
cheap Olsen. The same result updates the residual reference used by `--reltol` after every outer iteration.

The AUTO controls and defaults are `--ritz-stabilization-tolerance=1e-3`, `--auto-probe-interval=5`, and
`--auto-probe-length=3`.

## Warm Start

Save a partial cheap Olsen solution:

```bash
./build/Release/bench/standard/grit-bench-standard \
  --algo gdplusk \
  --matrix-path bench/data/finance256/finance256.mtx \
  --residual-correction=cheap-olsen \
  --abstol=1e-3 \
  --save-eigvec=bench/data/finance256/warmstart.h5
```

Use it as the initial guess in a later run:

```bash
./build/Release/bench/standard/grit-bench-standard \
  --algo gdplusk \
  --matrix-path bench/data/finance256/finance256.mtx \
  --initial-guess=bench/data/finance256/warmstart.h5 \
  --residual-correction=jacobi-davidson \
  --abstol=1e-5 \
  --refined-rayleigh-ritz
```

The benchmark now stores saved eigvecs under `/grit/standard/<algo>/eigvecs`. When loading an initial guess, it also
accepts the legacy dataset path
`/grit/standard/eigvecs`.

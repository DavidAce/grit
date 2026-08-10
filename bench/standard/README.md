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

With `--reltol` enabled, GRIT does not use the residual of the initial guess as its reference. Instead, it waits until
the selected Ritz values saturate around candidate eigenpairs and then records their current residual norms. The
relative target is `reltol * reference residual`, combined with the absolute target from `--abstol`. If the Ritz values
start progressing again, GRIT clears the references and records new ones when they saturate. Before a reference exists,
only `--abstol` applies.

The Ritz-saturation test starts after five history entries and checks every trailing history range down to the latest
five entries. The regression uses cumulative matrix-vector products as its coordinate, so its slope measures Ritz-value
drift per matvec. For pair `i`, the absolute drift-per-matvec threshold is

```text
max(ritz_saturation_tolerance * abs(lambda_i), epsilon * max(abs(T_evals)))
```

The Ritz value is progressing when the absolute fitted slope per matvec exceeds this threshold. The default saturation
tolerance is `1e-5`.

## AUTO Residual Correction

`--residual-correction=auto` starts with cheap Olsen. Once at least five Ritz-history entries are available, AUTO uses
the same Ritz-saturation test as solver termination. Cheap Olsen continues while any unconverged pair is still
progressing and switches to Jacobi-Davidson after `max(1, saturation_count_max / 2)` consecutive saturated iterations.
The absolute slope makes this independent of the selected Ritz mode.

Saturation counters continue across correction changes. If Jacobi-Davidson restores progress, the corresponding detector
resets its counter normally.

While JD is active, AUTO checks whether JD is still reducing each unconverged residual. For each Ritz pair, it fits a
least-squares line to the retained history of `log10(norm(r_i))`, using cumulative matrix-vector products from
consecutive JD iterations and at most the configured history size. At least two JD history entries are required. A
negative slope means that the residual is decreasing, so JD continues. A zero or positive slope for any unconverged pair
starts a probe of
`--auto-probe-length` consecutive cheap Olsen iterations.

`--auto-max-probes` limits the probes performed while the Ritz values remain in the same region. Zero disables probes
and
`-1` allows unlimited probes. After the minimum probe length, AUTO applies the same saturation test used for the initial
switch. A saturated probe is extended until the AUTO switch count is reached; a progressing probe continues with cheap
Olsen and resets the probe count.

The AUTO controls and defaults are `--ritz-saturation-tolerance=1e-5`, `--auto-probe-length=5`, and
`--auto-max-probes=1`.

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

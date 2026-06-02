# GRIT

`GRIT` is the Generalized Ritz Iteration Toolkit: a C++23 library for matrix-free
Ritz iteration eigensolvers. Operators are supplied as Eigen block callbacks.

## Requirements

  * CMake 3.24 or newer.
  * A C++23 compiler.
  * Eigen 3.4 or later, before Eigen 6.
  * fmt.
  * spdlog.
  * OpenMP.

The default build enables `double` and `std::complex<double>`. Other scalar
widths can be enabled with CMake options.

## Examples

Standard Hermitian problem:

```cpp
#include <grit/grit.h>

#include <Eigen/Core>
#include <print>

int main() {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(4, 4);
    A_matrix << 1.0, 0.1, 0.0, 0.0,
        0.1, 2.0, 0.2, 0.0,
        0.0, 0.2, 3.0, 0.3,
        0.0, 0.0, 0.3, 4.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](const auto &X) {
        return A_matrix * X;
    });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev              = 1;
    solver.config.ncv              = A_matrix.rows();
    solver.config.block_size       = 1;
    solver.config.ritz             = grit::OptRitz::SR;
    solver.set_initial_guess(Matrix::Identity(A_matrix.rows(), A_matrix.rows()));
    solver.run();

    auto view = grit::solver_view<double>(solver);
    for(Eigen::Index i = 0; i < view.eigVal().size(); ++i) {
        std::println("lambda[{}] = {:.16e}", i, view.eigVal()(i));
    }
}
```

Generalized Hermitian problem:

```cpp
#include <grit/grit.h>

#include <Eigen/Core>
#include <print>

int main() {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(4, 4);
    A_matrix << 1.0, 0.1, 0.0, 0.0,
        0.1, 2.0, 0.2, 0.0,
        0.0, 0.2, 3.0, 0.3,
        0.0, 0.0, 0.3, 4.0;

    Matrix B_matrix = A_matrix * A_matrix;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) {
        return A_matrix * X;
    });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) {
        return B_matrix * X;
    });

    grit::generalized::lanczos<double> solver(A, B);
    solver.config.nev              = 1;
    solver.config.ncv              = A_matrix.rows();
    solver.config.block_size       = 1;
    solver.config.ritz             = grit::OptRitz::LM;
    solver.set_initial_guess(Matrix::Identity(A_matrix.rows(), A_matrix.rows()));
    solver.run();

    auto view = grit::solver_view<double>(solver);
    std::println("generalized lambda = {:.16e}", view.eigVal()(0));
    std::println("recovered A eigenvalue = {:.16e}", 1.0 / view.eigVal()(0));
}
```

The same standard and generalized aliases are available for `gdplusk`,
`lanczos`, and `lobpcg`.

More complete examples are available in `examples/`. They can be built with an
existing preset:

```bash
cmake --preset release-conan -DGRIT_BUILD_EXAMPLES=ON
cmake --build --preset release-conan --target build-all-examples
```

## Install

```bash
git clone https://github.com/DavidAce/grit.git
cd grit
cmake --preset release-conan
cmake --build --preset release-conan
ctest --preset release-conan
cmake --install build/release-conan
```

Use `release-find` instead of `release-conan` when Eigen, fmt, spdlog, and
OpenMP are already visible to CMake.

## Presets

| Preset          | Dependency mode | Description                                              |
|:----------------|:----------------|:---------------------------------------------------------|
| `release-find`  | `find`          | Use dependencies already visible to `find_package`.      |
| `release-cmake` | `cmake`         | Build and install CMake-provided dependencies during configuration. |
| `release-conan` | `conan`         | Install dependencies with Conan during configuration.                |

For example:

```bash
cmake --preset release-conan
cmake --build --preset release-conan
ctest --preset release-conan
cmake --install build/release-conan
```

## Use From CMake

After installation:

```cmake
find_package(grit REQUIRED)

add_executable(my_program main.cpp)
target_link_libraries(my_program PRIVATE grit::grit)
```

## CMake Options

| Option                | Default | Description                                                                                                                    |
|:----------------------|:--------|:-------------------------------------------------------------------------------------------------------------------------------|
| `GRIT_ENABLE_32BIT`   | `OFF`   | Compile `float` and `std::complex<float>` instantiations.                                                                      |
| `GRIT_ENABLE_64BIT`   | `ON`    | Compile `double` and `std::complex<double>` instantiations.                                                                    |
| `GRIT_ENABLE_80BIT`   | `OFF`   | Compile `long double` and `std::complex<long double>` instantiations.                                                          |
| `GRIT_ENABLE_128BIT`  | `OFF`   | Compile `std::float128_t` and `std::complex<std::float128_t>` instantiations. Requires compiler support for `std::float128_t`. |
| `GRIT_ENABLE_TESTS`   | `OFF`   | Build the test programs and enable CTest.                                                                                      |
| `GRIT_BUILD_EXAMPLES` | `OFF`   | Build the example programs in `examples/`.                                                                                     |

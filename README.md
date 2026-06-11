
[![Ubuntu 24.04](https://github.com/DavidAce/grit/actions/workflows/ubuntu-24.04.yml/badge.svg)](https://github.com/DavidAce/grit/actions/workflows/ubuntu-24.04.yml)
[![codecov](https://codecov.io/github/DavidAce/grit/graph/badge.svg?token=PPTM8MBW52)](https://codecov.io/github/DavidAce/grit)

# GRIT
**Generalized Ritz Iteration Toolkit**

`GRIT` is a C++23 library for matrix-free iterative eigensolvers.

## Requirements
  * Symmetric or hermitian matrices
  * CMake 3.24 or newer
  * A C++23 compiler with support for OpenMP
  * Eigen 3.4 or later
  * spdlog (and fmt)

The default build enables `double` and `std::complex<double>`. Other scalar types can be enabled with CMake options.

## Examples

Standard Hermitian problem:

```cpp
#include <grit/grit.h>

#include <Eigen/Core>
#include <print>

int main() {
    // Find the smallest real eigenpair
    
    auto A_matrix = Eigen::MatrixXd(4, 4);
    A_matrix << 1.0, 0.1, 0.0, 0.0,
                0.1, 2.0, 0.2, 0.0,
                0.0, 0.2, 3.0, 0.3,
                0.0, 0.0, 0.3, 4.0;

    auto A_matvec = grit::matvec<double>(A_matrix.rows(), [&](const auto &X) {
        return A_matrix * X;
    });

    auto solver = grit::standard::gdplusk<double>(A_matvec);
    solver.config.nev              = 1;
    solver.config.ncv              = A_matrix.rows();
    solver.config.block_size       = 1;
    solver.config.ritz             = grit::Ritz::SR;
    solver.set_initial_guess(Matrix::Identity(A_matrix.rows(), A_matrix.rows()));
    solver.run();

    auto view = solver.get_result();
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
    // Find the eigenpair closest to 0


    auto A_matrix = Eigen::MatrixXd(4, 4);
    A_matrix << 1.0, 0.1, 0.0, 0.0,
                0.1, 2.0, 0.2, 0.0,
                0.0, 0.2, 3.0, 0.3,
                0.0, 0.0, 0.3, 4.0;

    Eigen::MatrixXd B_matrix = A_matrix * A_matrix; 

    auto A_matvec = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) {
        return A_matrix * X;
    });
    auto B_matvec = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) {
        return B_matrix * X;
    });

    grit::generalized::lanczos<double> solver(A_matvec, B_matvec);
    solver.config.nev              = 1;
    solver.config.ncv              = A_matrix.rows();
    solver.config.block_size       = 1;
    solver.config.ritz             = grit::Ritz::LM;
    solver.set_initial_guess(Matrix::Identity(A_matrix.rows(), A_matrix.rows()));
    solver.run();

    auto view = solver.get_result();
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

| Preset          | Dependency mode | Description                                                         |
|:----------------|:----------------|:--------------------------------------------------------------------|
| `release-find`  | `find`          | Use dependencies already visible to `find_package`.                 |
| `release-cmake` | `cmake`         | Build and install CMake-provided dependencies during configuration. |
| `release-conan` | `conan`         | Install dependencies with Conan during configuration.               |

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

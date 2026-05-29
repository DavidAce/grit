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

## Example

```cpp
#include <grit/grit.h>

#include <Eigen/Core>
#include <iostream>

int main() {
    Eigen::MatrixXd H(4, 4);
    H << 1, 0, 0, 0,
         0, 2, 0, 0,
         0, 0, 3, 0,
         0, 0, 0, 4;

    auto A = grit::matvec<double>(H.rows(), [&](auto const &X) {
        return H * X;
    });
    Eigen::MatrixXd V = Eigen::MatrixXd::Random(H.rows(), 2);
    grit::standard::gdplusk<double> solver(A);
    solver.config.nev = 2;
    solver.config.ncv = H.rows();
    solver.config.block_size = 1;
    solver.config.max_basis_blocks = H.rows();
    solver.config.ritz = grit::OptRitz::SR;
    solver.set_initial_guess(V);
    solver.run();

    auto view = grit::solver_view<double>(solver);
    std::cout << view.eigVal().transpose() << '\n';
}
```

More complete examples are available in `examples/`. They can be built with:

```bash
cmake -S . -B build/examples \
    -DCMAKE_BUILD_TYPE=Release \
    -DGRIT_BUILD_EXAMPLES=ON
cmake --build build/examples --target build-all-examples
```

## Install

```bash
git clone https://github.com/DavidAce/grit.git
cd grit
cmake -S . -B build/release \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/path/to/grit
cmake --build build/release
ctest --test-dir build/release
cmake --install build/release
```

This uses the default `find` dependency mode, so Eigen, fmt, spdlog, and OpenMP
must already be visible to CMake.

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

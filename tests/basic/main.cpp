#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <grit/grit.h>

#include <cmath>
#include <complex>
#include <type_traits>

TEST_CASE("scalar aliases are configured") {
    static_assert(std::is_same_v<fp64, double>);
    static_assert(std::is_same_v<cx64, std::complex<double>>);
}

TEST_CASE("identity operator converges") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(3, 3);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    Matrix V = Matrix::Random(3, 1);

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev        = 1;
    solver.config.ncv        = 3;
    solver.config.block_size = 1;
    solver.config.max_basis_blocks = 3;
    solver.config.ritz       = grit::OptRitz::SR;
    solver.config.max_iters  = 10;
    solver.set_initial_guess(V);
    solver.run();

    auto view = grit::solver_view<double>(solver);
    REQUIRE(grit::has_flag(view.stopReason(), grit::StopReason::converged));
    REQUIRE(std::abs(view.eigVal()(0) - 1.0) < 1e-12);
}

TEST_CASE("common config is read directly from solver.config") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(3, 3);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.tol = 3.25e-7;

    REQUIRE(solver.rNormTol(0) == Approx(3.25e-7));
}

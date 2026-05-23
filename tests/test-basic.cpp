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

    grit::standard::problem<double> problem(A);
    grit::gdplusk_config<double>    cfg;
    cfg.nev       = 1;
    cfg.ncv       = 3;
    cfg.ritz      = grit::OptRitz::SR;
    cfg.max_iters = 10;
    cfg.set_initial_guess(V);

    grit::standard::gdplusk<double> solver(problem, cfg);
    solver.run();

    auto result = solver.result();
    REQUIRE(result.stopReason() == grit::StopReason::converged);
    REQUIRE(std::abs(result.eigVal()(0) - 1.0) < 1e-12);
}

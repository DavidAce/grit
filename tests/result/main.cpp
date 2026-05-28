#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <grit/grit.h>

TEST_CASE("solver result can be cleared") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(3, 3);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    Matrix V = Matrix::Random(3, 1);

    grit::standard::problem<double> problem(A);
    grit::gdplusk_config<double>    cfg;
    cfg.nev = 1;
    cfg.ncv = 3;
    cfg.block_size   = 1;
    cfg.set_initial_guess(V);

    grit::standard::gdplusk<double> solver(problem, cfg);
    solver.run();
    REQUIRE(solver.result().eigVal().size() == 1);

    solver.clear_result();
    REQUIRE(solver.result().eigVal().size() == 0);
}

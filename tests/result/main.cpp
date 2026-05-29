#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <grit/grit.h>

TEST_CASE("solver view reflects clear_result") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(3, 3);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    Matrix V = Matrix::Random(3, 1);

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev              = 1;
    solver.config.ncv              = 3;
    solver.config.block_size       = 1;
    solver.config.max_basis_blocks = 3;
    solver.set_initial_guess(V);
    solver.run();
    REQUIRE(grit::solver_view<double>(solver).eigVal().size() == 1);

    solver.clear_result();
    REQUIRE(grit::solver_view<double>(solver).eigVal().size() == 0);
}

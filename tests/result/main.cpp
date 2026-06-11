#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include <grit/grit.h>

TEST_CASE("solver run is reentrant") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(3, 3);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    Matrix V = Matrix::Random(3, 1);

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev        = 1;
    solver.config.ncv        = 3;
    solver.config.block_size = 1;
    solver.set_initial_guess(V);
    solver.run();
    auto first_view    = solver.get_result();
    auto first_iters   = first_view.iter();
    auto first_matvecs = first_view.num_matvecs_total();
    REQUIRE(first_view.eigVal().size() == 1);
    REQUIRE(grit::has_flag(first_view.stopReason(), grit::StopReason::converged));

    solver.run();
    auto view = solver.get_result();
    REQUIRE(view.eigVal().size() == 1);
    REQUIRE(grit::has_flag(view.stopReason(), grit::StopReason::converged));
    REQUIRE(view.iter() > first_iters);
    REQUIRE(view.num_matvecs_total() > first_matvecs);
}

int main(int argc, char **argv) {
    Catch::Session session;
    const int      return_code = session.applyCommandLine(argc, argv);
    if(return_code != 0) return return_code;
    return session.run();
}

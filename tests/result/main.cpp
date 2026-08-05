#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include <grit/grit.h>

namespace {
    template<typename Fn>
    void with_solved_identity_solver(Fn &&fn) {
        using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

        Matrix A_matrix = Matrix::Identity(3, 3);
        auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

        grit::standard::gdplusk<double> solver(A);
        solver.config.nev        = 1;
        solver.config.ncv        = 3;
        solver.config.block_size = 1;
        solver.config.ritz       = grit::Ritz::SR;
        solver.set_initial_guess(Matrix::Random(3, 1));
        solver.run();
        fn(solver);
    }
}

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
    auto first_result      = solver.get_result();
    auto first_outer_iters = first_result.outer_iter();
    auto first_matvecs     = first_result.num_matvecs_total();
    auto first_time        = first_result.time();
    REQUIRE(first_result.eigVal().size() == 1);
    REQUIRE(grit::has_flag(first_result.stopReason(), grit::StopReason::converged));

    solver.run();
    auto result = solver.get_result();
    REQUIRE(result.eigVal().size() == 1);
    REQUIRE(grit::has_flag(result.stopReason(), grit::StopReason::converged));
    REQUIRE(result.outer_iter() > first_outer_iters);
    REQUIRE(result.num_matvecs_total() > first_matvecs);
    REQUIRE(result.time() > first_time);
    REQUIRE(result.num_matvecs_a_total() > first_result.num_matvecs_a_total());
}

TEST_CASE("get_result returns an owning copy") {
    with_solved_identity_solver([](auto &solver) {
        auto view                = solver.get_result_view();
        auto original_eigval     = view.eigVal();
        auto original_eigvecs    = view.eigVecs();
        auto original_rnorms     = view.rNormsAbs();
        auto original_stopReason = view.stopReason();

        auto result = solver.get_result();
        result.eigVal().setConstant(-7.0);
        result.eigVecs().setZero();
        result.rNormsAbs().setConstant(11.0);

        REQUIRE((view.eigVal() - original_eigval).norm() == Approx(0.0));
        REQUIRE((view.eigVecs() - original_eigvecs).norm() == Approx(0.0));
        REQUIRE((view.rNormsAbs() - original_rnorms).norm() == Approx(0.0));
        REQUIRE(view.stopReason() == original_stopReason);
        REQUIRE((result.eigVecs() - view.eigVecs()).norm() > 0.1);
    });
}

TEST_CASE("results contain only requested eigenpairs") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(8, 8);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    for(Eigen::Index nev = 1; nev <= 2; ++nev) {
        grit::standard::gdplusk<double> solver(A);
        solver.config.nev        = nev;
        solver.config.ncv        = 8;
        solver.config.block_size = 2 * nev;
        solver.set_initial_guess(Matrix::Random(8, solver.config.block_size));
        solver.run();

        auto view = solver.get_result_view();
        REQUIRE(view.eigVal().size() == nev);
        REQUIRE(view.eigVecs().cols() == nev);
        REQUIRE(view.rNormsAbs().size() == nev);

        auto result = solver.get_result();
        REQUIRE(result.eigVal().size() == nev);
        REQUIRE(result.eigVecs().cols() == nev);
        REQUIRE(result.rNormsAbs().size() == nev);
    }
}

TEST_CASE("result view can be copied explicitly") {
    with_solved_identity_solver([](auto &solver) {
        auto view   = solver.get_result_view();
        auto result = view.to_result();

        REQUIRE((result.eigVal() - view.eigVal()).norm() == Approx(0.0));
        REQUIRE((result.eigVecs() - view.eigVecs()).norm() == Approx(0.0));
        REQUIRE((result.rNormsAbs() - view.rNormsAbs()).norm() == Approx(0.0));
        REQUIRE(result.stopReason() == view.stopReason());
        REQUIRE(result.num_matvecs_a_total() == view.num_matvecs_a_total());
        REQUIRE(result.time_build() == Approx(view.time_build()));
    });
}

TEST_CASE("owning result supports downstream eigenvector transforms") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    with_solved_identity_solver([](auto &solver) {
        auto view   = solver.get_result_view();
        auto result = solver.get_result();

        Matrix basis = Matrix::Identity(result.eigVecs().rows(), result.eigVecs().rows());
        basis(0, 0)  = -1.0;
        result.eigVecs() = basis * result.eigVecs();

        REQUIRE(result.eigVal()(0) == Approx(view.eigVal()(0)));
        REQUIRE((result.eigVecs() - view.eigVecs()).norm() > 0.1);
    });
}

int main(int argc, char **argv) {
    Catch::Session session;
    const int      return_code = session.applyCommandLine(argc, argv);
    if(return_code != 0) return return_code;
    return session.run();
}

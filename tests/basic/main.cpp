#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include <cmath>
#include <complex>
#include <grit/grit.h>
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
    solver.config.ritz       = grit::Ritz::SR;
    solver.config.max_iters  = 10;
    solver.set_initial_guess(V);
    solver.run();

    auto view = solver.get_result();
    REQUIRE(grit::has_flag(view.stopReason(), grit::StopReason::converged));
    REQUIRE(std::abs(view.eigVal()(0) - 1.0) < 1e-12);
}

TEST_CASE("common config is read directly from solver.config") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(3, 3);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.abstol = 3.25e-7;

    REQUIRE(solver.rNormAbsTarget(0) == Approx(3.25e-7));
}

TEST_CASE("residual tolerance uses the larger floor") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(3, 3);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev                          = 1;
    solver.config.use_rescaled_rnorm_tolerance = true;
    solver.config.abstol                       = 1e-5;
    solver.config.reltol                       = 1e-1;

    solver.status.eigVal              = Eigen::VectorXd::Constant(1, 1.0);
    solver.status.op_norm_estimate    = 10.0;
    solver.status.rnorm_abs_reference = Eigen::VectorXd::Zero(1);

    REQUIRE(solver.rNormAbsTarget(0) == Approx(1.0e-4));

    solver.status.rnorm_abs_reference = Eigen::VectorXd::Constant(1, 2.0e-3);

    REQUIRE(solver.rNormAbsTarget(0) == Approx(2.0e-4));
}

TEST_CASE("relative residual references use Ritz saturation") {
    using Matrix     = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorReal = Eigen::Vector<double, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev                       = 2;
    solver.config.abstol                    = 1e-12;
    solver.config.reltol                    = 1e-1;
    solver.config.ritz_saturation_tolerance = 1e-3;
    solver.V                                = Matrix::Identity(4, 2);
    solver.status.optIdx                    = {0, 1};
    solver.status.eigVal                    = (VectorReal(2) << 1.0, 1.0).finished();
    solver.status.rNormsAbs                 = VectorReal::Ones(2);
    solver.status.rnorm_abs_reference       = VectorReal::Zero(2);
    solver.status.num_matvecs               = 1;

    for(Eigen::Index iter = 0; iter < 5; ++iter) {
        solver.T_evals          = (VectorReal(2) << 1.0, 2.0 + static_cast<double>(iter)).finished();
        solver.status.rNormsAbs = (VectorReal(2) << 0.5, 1.0).finished();
        solver.Base::updateStatus();
    }

    REQUIRE(solver.status.rnorm_abs_reference(0) == Approx(0.0));
    REQUIRE(solver.status.rnorm_abs_reference(1) == Approx(0.0));

    for(Eigen::Index iter = 0; iter < 9; ++iter) {
        solver.T_evals = (VectorReal(2) << 1.0, 7.0).finished();
        solver.Base::updateStatus();
    }

    REQUIRE(solver.status.rnorm_abs_reference(0) == Approx(0.5));
    REQUIRE(solver.status.rnorm_abs_reference(1) == Approx(1.0));

    solver.T_evals = (VectorReal(2) << 1.1, 7.0).finished();
    solver.Base::updateStatus();

    REQUIRE(solver.status.rnorm_abs_reference(0) == Approx(0.0));
    REQUIRE(solver.status.rnorm_abs_reference(1) == Approx(1.0));
}

int main(int argc, char **argv) {
    Catch::Session session;
    const int      return_code = session.applyCommandLine(argc, argv);
    if(return_code != 0) return return_code;
    return session.run();
}

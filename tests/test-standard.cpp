#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <grit/grit.h>

#include <Eigen/Eigenvalues>
#include <cmath>

namespace {
    template<typename VecA, typename VecB>
    void require_close(const VecA &a, const VecB &b, double tol) {
        REQUIRE(a.size() == b.size());
        for(Eigen::Index i = 0; i < a.size(); ++i) REQUIRE(std::abs(a(i) - b(i)) < tol);
    }
}

TEST_CASE("standard gdplusk matches dense eigensolver") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 3.0, 0.5, 0.0, 0.0,
        0.0, 0.5, 2.0, 0.25, 0.0,
        0.0, 0.0, 0.25, 5.0, 0.5,
        0.0, 0.0, 0.0, 0.5, 6.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    Matrix V = Matrix::Identity(A_matrix.rows(), A_matrix.rows());

    grit::standard::problem<double> problem(A);
    grit::gdplusk_config<double>    cfg;
    cfg.nev       = 2;
    cfg.ncv       = A_matrix.rows();
    cfg.b         = 1;
    cfg.ritz      = grit::OptRitz::SR;
    cfg.max_iters = 20;
    cfg.set_initial_guess(V);

    grit::standard::gdplusk<double> solver(problem, cfg);
    solver.run();

    Eigen::SelfAdjointEigenSolver<Matrix> exact(A_matrix);
    auto                                  result = solver.result();
    REQUIRE(result.stopReason() == grit::StopReason::converged);
    require_close(result.eigVal(), exact.eigenvalues().head(2), 1e-10);
}

TEST_CASE("standard jacobi-davidson correction invokes preconditioner callbacks") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = grit::MatVec<double>::VectorType;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 3.0, 0.5, 0.0, 0.0,
        0.0, 0.5, 2.0, 0.25, 0.0,
        0.0, 0.0, 0.25, 5.0, 0.5,
        0.0, 0.0, 0.0, 0.5, 6.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    int update_count = 0;
    int apply_count  = 0;
    A.set_preconditioner_update([&](double) { update_count++; });
    A.set_preconditioner_apply([&](const Eigen::Ref<const Vector> &x, Eigen::Ref<Vector> y, double) {
        apply_count++;
        y = x;
    });

    Matrix V(A_matrix.rows(), 3);
    V << 1.0, 0.2, 0.3,
        0.4, 1.0, 0.2,
        0.3, 0.5, 1.0,
        0.2, 0.4, 0.5,
        0.5, 0.3, 0.4;

    grit::standard::problem<double> problem(A);
    grit::gdplusk_config<double>    cfg;
    cfg.nev         = 1;
    cfg.ncv         = 3;
    cfg.b           = 1;
    cfg.ritz        = grit::OptRitz::SR;
    cfg.max_iters   = 5;
    cfg.inner_iters = 20;
    cfg.inner_tol   = 1e-8;
    cfg.set_initial_guess(V);

    grit::standard::gdplusk<double> solver(problem, cfg);
    solver.residual_correction_type = grit::form::base<double>::ResidualCorrectionType::JACOBI_DAVIDSON;
    solver.run();

    REQUIRE(update_count > 0);
    REQUIRE(apply_count > 0);
}

TEST_CASE("standard jacobi-davidson correction defaults to identity preconditioner") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 3.0, 0.5, 0.0, 0.0,
        0.0, 0.5, 2.0, 0.25, 0.0,
        0.0, 0.0, 0.25, 5.0, 0.5,
        0.0, 0.0, 0.0, 0.5, 6.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    Matrix V(A_matrix.rows(), 3);
    V << 1.0, 0.2, 0.3,
        0.4, 1.0, 0.2,
        0.3, 0.5, 1.0,
        0.2, 0.4, 0.5,
        0.5, 0.3, 0.4;

    grit::standard::problem<double> problem(A);
    grit::gdplusk_config<double>    cfg;
    cfg.nev         = 1;
    cfg.ncv         = 3;
    cfg.b           = 1;
    cfg.ritz        = grit::OptRitz::SR;
    cfg.max_iters   = 2;
    cfg.inner_iters = 20;
    cfg.inner_tol   = 1e-8;
    cfg.set_initial_guess(V);

    grit::standard::gdplusk<double> solver(problem, cfg);
    solver.residual_correction_type = grit::form::base<double>::ResidualCorrectionType::JACOBI_DAVIDSON;

    REQUIRE_NOTHROW(solver.run());
}

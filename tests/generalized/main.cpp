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

TEST_CASE("generalized gdplusk matches dense eigensolver") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 3.0, 0.5, 0.0, 0.0,
        0.0, 0.5, 2.0, 0.25, 0.0,
        0.0, 0.0, 0.25, 5.0, 0.5,
        0.0, 0.0, 0.0, 0.5, 6.0;

    Matrix B_matrix = Matrix::Identity(5, 5);
    B_matrix.diagonal() << 1.0, 1.5, 2.0, 2.5, 3.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    Matrix V = Matrix::Identity(A_matrix.rows(), A_matrix.rows());

    grit::generalized::gdplusk<double> solver(A, B);
    solver.config.nev              = 1;
    solver.config.ncv              = A_matrix.rows();
    solver.config.block_size       = 1;
    solver.config.max_basis_blocks = A_matrix.rows();
    solver.config.ritz             = grit::OptRitz::SR;
    solver.config.max_iters        = 20;
    solver.set_initial_guess(V);
    solver.run();

    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> exact(A_matrix, B_matrix);
    auto view = grit::solver_view<double>(solver);
    REQUIRE(view.stopReason() == grit::StopReason::converged);
    require_close(view.eigVal(), exact.eigenvalues().head(1), 1e-10);
}

TEST_CASE("generalized lanczos matches dense eigensolver") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 3.0, 0.5, 0.0, 0.0,
        0.0, 0.5, 2.0, 0.25, 0.0,
        0.0, 0.0, 0.25, 5.0, 0.5,
        0.0, 0.0, 0.0, 0.5, 6.0;

    Matrix B_matrix = Matrix::Identity(5, 5);
    B_matrix.diagonal() << 1.0, 1.5, 2.0, 2.5, 3.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    Matrix V = Matrix::Identity(A_matrix.rows(), A_matrix.rows());

    grit::generalized::lanczos<double> solver(A, B);
    solver.config.nev              = 2;
    solver.config.ncv              = A_matrix.rows();
    solver.config.block_size       = 1;
    solver.config.max_basis_blocks = A_matrix.rows();
    solver.config.ritz             = grit::OptRitz::SR;
    solver.config.max_iters        = 20;
    solver.set_initial_guess(V);
    solver.run();

    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> exact(A_matrix, B_matrix);
    auto view = grit::solver_view<double>(solver);
    REQUIRE(view.stopReason() == grit::StopReason::converged);
    require_close(view.eigVal(), exact.eigenvalues().head(2), 1e-10);
}

TEST_CASE("generalized lobpcg matches dense eigensolver") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 3.0, 0.5, 0.0, 0.0,
        0.0, 0.5, 2.0, 0.25, 0.0,
        0.0, 0.0, 0.25, 5.0, 0.5,
        0.0, 0.0, 0.0, 0.5, 6.0;

    Matrix B_matrix = Matrix::Identity(5, 5);
    B_matrix.diagonal() << 1.0, 1.5, 2.0, 2.5, 3.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    Matrix V = Matrix::Identity(A_matrix.rows(), A_matrix.rows());

    grit::generalized::lobpcg<double> solver(A, B);
    solver.config.nev                    = 1;
    solver.config.ncv                    = A_matrix.rows();
    solver.config.block_size             = 1;
    solver.config.max_basis_blocks       = A_matrix.rows();
    solver.config.max_extra_ritz_history = 1;
    solver.config.max_ritz_residual_history = 1;
    solver.config.ritz                   = grit::OptRitz::SR;
    solver.config.max_iters              = 20;
    solver.set_initial_guess(V);
    solver.run();

    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> exact(A_matrix, B_matrix);
    auto view = grit::solver_view<double>(solver);
    REQUIRE(view.stopReason() == grit::StopReason::converged);
    require_close(view.eigVal(), exact.eigenvalues().head(1), 1e-10);
}

TEST_CASE("generalized jacobi-davidson b-only correction supports l2 and bm projectors") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 3.0, 0.5, 0.0, 0.0,
        0.0, 0.5, 2.0, 0.25, 0.0,
        0.0, 0.0, 0.25, 5.0, 0.5,
        0.0, 0.0, 0.0, 0.5, 6.0;

    Matrix B_matrix = Matrix::Identity(5, 5);
    B_matrix.diagonal() << 1.0, 1.5, 2.0, 2.5, 3.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    Matrix V = Matrix::Identity(A_matrix.rows(), A_matrix.rows());

    grit::generalized::gdplusk<double>::Config cfg;
    cfg.nev              = 1;
    cfg.ncv              = A_matrix.rows();
    cfg.block_size       = 1;
    cfg.max_basis_blocks = A_matrix.rows();
    cfg.ritz             = grit::OptRitz::SR;
    cfg.max_iters        = 2;
    cfg.inner_max_iters  = 20;
    cfg.inner_tol        = 1e-8;

    SECTION("l2 projectors") {
        auto                             section_cfg = cfg;
        section_cfg.residual_correction_type        = grit::ResidualCorrectionType::JACOBI_DAVIDSON;
        section_cfg.use_jd_b_only                   = true;
        section_cfg.use_b_inner_product             = false;
        grit::generalized::gdplusk<double> solver(A, B);
        solver.config = section_cfg;
        solver.set_initial_guess(V);

        REQUIRE_NOTHROW(solver.run());
    }

    SECTION("bm projectors") {
        auto                             section_cfg = cfg;
        section_cfg.residual_correction_type        = grit::ResidualCorrectionType::JACOBI_DAVIDSON;
        section_cfg.use_jd_b_only                   = true;
        section_cfg.use_b_inner_product             = true;
        grit::generalized::gdplusk<double> solver(A, B);
        solver.config = section_cfg;
        solver.set_initial_guess(V);

        REQUIRE_NOTHROW(solver.run());
    }
}

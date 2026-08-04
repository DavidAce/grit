#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include "solver_test_utils.h"
#include <cmath>
#include <cstdlib>
#include <Eigen/Eigenvalues>
#include <format>
#include <grit/grit.h>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace {
    int test_log_fd = STDERR_FILENO;

    void write_test_log(const std::string &message) {
        const char *data = message.data();
        auto        size = message.size();
        while(size != 0) {
            const auto written = ::write(test_log_fd, data, size);
            if(written <= 0) return;
            data += written;
            size -= static_cast<std::size_t>(written);
        }
    }

    template<typename VecA, typename VecB>
    void require_close(const VecA &a, const VecB &b, double abstol) {
        REQUIRE(a.size() == b.size());
        for(Eigen::Index i = 0; i < a.size(); ++i) REQUIRE(std::abs(a(i) - b(i)) < abstol);
    }

    template<typename VecA, typename VecB>
    void print_eigenvalue_comparison(std::string_view label, const VecA &computed, const VecB &exact, Eigen::Index count) {
        write_test_log(std::format("{} eigenvalue comparison:\n", label));
        for(Eigen::Index i = 0; i < count; ++i) {
            const auto diff = std::abs(computed(i) - exact(i));
            write_test_log(std::format("  [{}] computed {:.16e} exact {:.16e} abs_diff {:.3e}\n", i, computed(i), exact(i), diff));
        }
    }
}

TEST_CASE("generalized refined Rayleigh-Ritz reports the Rayleigh quotient of the refined vector") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    std::srand(1);

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.0, 0.0, 0.5, 2.0, 0.25, 0.0, 0.0, 0.0, 0.25, 5.0, 0.5, 0.0, 0.0, 0.0, 0.5, 6.0;

    Matrix B_matrix = Matrix::Identity(5, 5);
    B_matrix.diagonal() << 1.0, 1.5, 2.0, 2.5, 3.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    grit::generalized::gdplusk<double> solver(A, B);
    solver.config.nev                                     = 1;
    solver.config.ncv                                     = A_matrix.rows();
    solver.config.block_size                              = 1;
    solver.config.maxRetainBlocks                         = 1;
    solver.config.ritz                                    = grit::Ritz::SR;
    solver.config.max_iters                               = 20;
    solver.config.abstol                                  = 1e-14;
    solver.config.residual_correction_type                = grit::ResidualCorrectionType::CHEAP_OLSEN;
    solver.config.use_refined_rayleigh_ritz               = true;
    solver.config.use_rayleigh_quotients_instead_of_evals = true;
    solver.set_initial_guess(grit_test::seeded_initial_guess<double>(A_matrix.rows(), solver.config.block_size, 71));
    solver.run();

    auto view = solver.get_result();
    REQUIRE(view.eigVal().size() == 1);
    REQUIRE(view.eigVecs().cols() >= 1);

    const auto v  = view.eigVecs().col(0);
    const auto av = A_matrix * v;
    const auto bv = B_matrix * v;
    const auto rq = std::real(v.dot(av)) / std::real(v.dot(bv));
    REQUIRE(std::abs(view.eigVal()(0) - rq) < 1e-12);
}

TEST_CASE("generalized refined Ritz vector remains B-normalized with bm projector") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(8, 8);
    A_matrix << 9.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 7.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 6.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 5.0, 0.3,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 4.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 3.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 2.5, 0.05, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.05, 2.0;

    Matrix B_matrix = Matrix::Identity(8, 8);
    B_matrix.diagonal() << 1.0, 1.2, 1.7, 2.3, 3.1, 4.2, 5.6, 7.4;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    grit::generalized::gdplusk<double> solver(A, B);
    solver.config.nev                                     = 1;
    solver.config.ncv                                     = 8;
    solver.config.block_size                              = 1;
    solver.config.maxRetainBlocks                         = 2;
    solver.config.maxPrevBlocks                           = 2;
    solver.config.ritz                                    = grit::Ritz::LM;
    solver.config.max_iters                               = 100;
    solver.config.abstol                                  = 1e-12;
    solver.config.inner_max_iters                         = 40;
    solver.config.inner_tol                               = 1e-8;
    solver.config.use_b_inner_product                     = true;
    solver.config.use_refined_rayleigh_ritz               = true;
    solver.config.use_rayleigh_quotients_instead_of_evals = true;
    solver.config.use_jd_b_only                           = true;
    solver.config.residual_correction_type                = grit::ResidualCorrectionType::JACOBI_DAVIDSON;
    solver.config.user_callback                           = [&](const auto &s) {
        const Eigen::Index active_ritz_cols = std::min<Eigen::Index>(s.config.block_size, s.V.cols());
        if(active_ritz_cols == 0) return;

        REQUIRE(s.K.cols() <= s.config.maxPrevBlocks * s.config.block_size);

        const auto V_active  = s.V.leftCols(active_ritz_cols);
        const auto BV_active = s.BV.leftCols(active_ritz_cols);
        Matrix     G_stored  = V_active.adjoint() * BV_active;
        Matrix     G_fresh   = V_active.adjoint() * (B_matrix * V_active);
        Matrix     I         = Matrix::Identity(active_ritz_cols, active_ritz_cols);
        G_stored             = (G_stored + G_stored.adjoint()) * 0.5;
        G_fresh              = (G_fresh + G_fresh.adjoint()) * 0.5;

        REQUIRE((G_stored - I).norm() < 1e-10);
        REQUIRE((G_fresh - I).norm() < 1e-10);
    };
    solver.set_initial_guess(grit_test::seeded_initial_guess<double>(A_matrix.rows(), solver.config.block_size, 72));

    REQUIRE_NOTHROW(solver.run());
    auto view = solver.get_result();
    REQUIRE(view.eigVecs().cols() >= 1);

    const auto v      = view.eigVecs().col(0);
    const auto b_norm = std::real(v.dot(B_matrix * v));
    REQUIRE(std::abs(b_norm - 1.0) < 1e-10);
}

TEST_CASE("bm orthogonalization removes the one-sided B projection") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    constexpr double skew = 1e-5;
    auto             A    = grit::matvec<double>(2, [](auto const &X) { return X; });
    auto             B    = grit::matvec<double>(2, [skew](auto const &X) {
        Matrix BX          = X;
        BX.row(0).array() += skew * X.row(1).array();
        return BX;
    });

    grit::generalized::gdplusk<double> solver(A, B);
    solver.config.use_b_inner_product = true;
    solver.config.log_level           = spdlog::level::off;
    if(solver.log) solver.log->set_level(spdlog::level::off);

    Matrix X(2, 1);
    X << 1.0, 0.0;
    Matrix AX = X;
    Matrix BX = X;

    Matrix Y(2, 1);
    Y << 0.0, 1.0;
    Matrix AY = Y;
    Matrix BY = Matrix::Zero(2, 1);

    typename decltype(solver)::OrthMeta m;
    m.refresh_by = true;

    REQUIRE_NOTHROW(solver.block_bm_orthogonalize(X, AX, BX, Y, AY, BY, m, decltype(solver)::Base::RefreshMult::NO));
    REQUIRE(std::abs((X.adjoint() * BY)(0, 0)) < 1e-12);
    REQUIRE(std::abs((BX.adjoint() * Y)(0, 0)) > 1e-7);

    typename decltype(solver)::OrthMeta meta;
    meta.analyze_bm_orthogonality(X, BX, Y, BY);
    REQUIRE(meta.Gram.norm() < 1e-12);
    REQUIRE(meta.orthError < 1e-4);
    REQUIRE(meta.orthError > 1e-7);
    REQUIRE(meta.skewError > 1e-7);
}

TEST_CASE("generalized gdplusk matches dense eigensolver") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.0, 0.0, 0.5, 2.0, 0.25, 0.0, 0.0, 0.0, 0.25, 5.0, 0.5, 0.0, 0.0, 0.0, 0.5, 6.0;

    Matrix B_matrix = Matrix::Identity(5, 5);
    B_matrix.diagonal() << 1.0, 1.5, 2.0, 2.5, 3.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    grit::generalized::gdplusk<double> solver(A, B);
    solver.config.nev        = 1;
    solver.config.ncv        = A_matrix.rows();
    solver.config.block_size = 1;
    solver.config.ritz       = grit::Ritz::SR;
    solver.config.max_iters  = 20;
    solver.set_initial_guess(Matrix::Identity(A_matrix.rows(), A_matrix.rows()));
    solver.run();

    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> exact(A_matrix, B_matrix);
    auto                                             view = solver.get_result();
    REQUIRE(view.stopReason() == grit::StopReason::converged);
    print_eigenvalue_comparison("generalized gdplusk", view.eigVal(), exact.eigenvalues(), view.eigVal().size());
    require_close(view.eigVal(), exact.eigenvalues().head(1), 1e-10);
}

TEST_CASE("generalized gdplusk converges with l2 and bm projectors") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.0, 0.0, 0.5, 2.0, 0.25, 0.0, 0.0, 0.0, 0.25, 5.0, 0.5, 0.0, 0.0, 0.0, 0.5, 6.0;

    Matrix B_matrix = Matrix::Identity(5, 5);
    B_matrix.diagonal() << 1.0, 1.5, 2.0, 2.5, 3.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> exact(A_matrix, B_matrix);
    for(bool use_b_inner_product : {false, true}) {
        grit::generalized::gdplusk<double> solver(A, B);
        solver.config.nev                      = 1;
        solver.config.ncv                      = A_matrix.rows();
        solver.config.block_size               = 1;
        solver.config.ritz                     = grit::Ritz::SR;
        solver.config.max_iters                = 20;
        solver.config.abstol                   = 1e-12;
        solver.config.use_b_inner_product      = use_b_inner_product;
        solver.config.residual_correction_type = grit::ResidualCorrectionType::CHEAP_OLSEN;
        solver.set_initial_guess(grit_test::seeded_initial_guess<double>(A_matrix.rows(), solver.config.block_size, 54));
        solver.run();

        auto view = solver.get_result();
        print_eigenvalue_comparison(use_b_inner_product ? "generalized gdplusk bm projector" : "generalized gdplusk l2 projector", view.eigVal(),
                                    exact.eigenvalues(), view.eigVal().size());
        REQUIRE(view.stopReason() == grit::StopReason::converged);
        REQUIRE(view.rNormsAbs()(0) < 1e-10);
        require_close(view.eigVal(), exact.eigenvalues().head(1), 1e-10);
    }
}

TEST_CASE("B-metric orthonormality check does not throw on skew-only Gram error") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(2, 2);
    Matrix B_matrix = Matrix::Identity(2, 2);

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    grit::generalized::gdplusk<double> solver(A, B);
    solver.config.use_b_inner_product = true;
    solver.setLogger(spdlog::level::off);

    Matrix X = Matrix::Identity(2, 2);
    Matrix B_X(2, 2);
    B_X << 1.0, 2.0, -2.0, 1.0;

    grit::generalized::gdplusk<double>::OrthMeta meta;
    REQUIRE_NOTHROW(solver.assert_bm_orthonormal(X, B_X, meta));
}

TEST_CASE("B-metric orthogonality check accepts cancellation-scaled defects") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    constexpr double op_scale = 1e4;
    Matrix           A_matrix = Matrix::Identity(4, 4);
    Matrix           B_matrix = op_scale * Matrix::Identity(4, 4);

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });
    B.set_op_norm(op_scale);

    grit::generalized::gdplusk<double> solver(A, B);
    solver.config.use_b_inner_product = true;
    solver.setLogger(spdlog::level::off);

    Matrix X  = Matrix::Zero(4, 1);
    Matrix BX = Matrix::Zero(4, 1);
    Matrix BY = Matrix::Zero(4, 1);
    X(0, 0)   = 2.0e4;
    BX(0, 0)  = 1.0 / X(0, 0);
    BY(0, 0)  = 5.0e-13;
    BY(1, 0)  = 2.0e-4;

    Matrix Y = Matrix::Zero(4, 1);
    Y(0, 0)  = 1.0e4;

    const double gamma_n =
        static_cast<double>(X.rows()) * std::numeric_limits<double>::epsilon() / (1.0 - static_cast<double>(X.rows()) * std::numeric_limits<double>::epsilon());
    const double cancellation_multiplier = solver.bm_cancellation_multiplier(Y, BY);
    const double dotTol                  = gamma_n * X.norm() * BY.norm();
    const double scaledTol               = 10.0 * std::max(solver.orthTol, dotTol * cancellation_multiplier);
    const double oldTol                  = 10.0 * std::max({solver.orthTol * static_cast<double>(X.cols()) * (X.norm() + BY.norm()), solver.orthTol,
                                                            solver.orthTol * static_cast<double>(X.cols()) * op_scale});
    const double orthError               = (X.adjoint() * BY).norm();

    grit::generalized::gdplusk<double>::OrthMeta meta;
    REQUIRE(orthError > oldTol);
    REQUIRE(orthError < scaledTol);
    REQUIRE_NOTHROW(solver.assert_bm_orthogonal(X, BX, Y, BY, meta));
}

TEST_CASE("B-metric orthonormality check scales by local Rayleigh quotient") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    constexpr double op_scale = 1e6;
    Matrix           A_matrix = Matrix::Identity(4, 4);
    Matrix           B_matrix = op_scale * Matrix::Identity(4, 4);

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });
    B.set_op_norm(op_scale);

    grit::generalized::gdplusk<double> solver(A, B);
    solver.config.use_b_inner_product = true;
    solver.setLogger(spdlog::level::off);

    Matrix X = Matrix::Zero(4, 1);
    X(0, 0)  = 1.0e4;

    Matrix B_X = Matrix::Zero(4, 1);
    B_X(0, 0)  = (1.0 + 1.0e-8) / X(0, 0);

    grit::generalized::gdplusk<double>::OrthMeta meta;

    const double gamma_n =
        static_cast<double>(X.rows()) * std::numeric_limits<double>::epsilon() / (1.0 - static_cast<double>(X.rows()) * std::numeric_limits<double>::epsilon());
    const double rq                      = std::abs(X.col(0).dot(B_X.col(0))) / X.col(0).squaredNorm();
    const double cancellation_multiplier = std::clamp(op_scale / rq, 1.0, 1.0 / std::sqrt(std::numeric_limits<double>::epsilon()));
    const double dotTol                  = gamma_n * X.norm() * B_X.norm();
    const double scaledTol               = 10.0 * std::max(solver.orthTol, dotTol * cancellation_multiplier);
    const double symmError               = std::abs((X.adjoint() * B_X)(0, 0) - 1.0);

    REQUIRE(cancellation_multiplier > 1.0);
    REQUIRE(symmError < scaledTol);
    REQUIRE_NOTHROW(solver.assert_bm_orthonormal(X, B_X, meta));

    B_X(0, 0) = 1.02 / X(0, 0);
    REQUIRE_THROWS(solver.assert_bm_orthonormal(X, B_X, meta));
}

TEST_CASE("B-metric eig orthonormalizer compresses and normalizes dependent blocks") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(4, 4);
    Matrix B_matrix = Matrix::Identity(4, 4);
    B_matrix.diagonal() << 1.0, 2.0, 3.0, 4.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    grit::generalized::gdplusk<double> solver(A, B);
    solver.config.use_b_inner_product = true;

    Matrix Y(4, 3);
    Y << 1.0, 1.0, 0.0, 0.0, 1e-12, 1.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0;
    Matrix AY;
    Matrix BY = B_matrix * Y;

    grit::generalized::gdplusk<double>::OrthMeta meta;
    meta.maskPolicy = grit::generalized::gdplusk<double>::Base::MaskPolicy::COMPRESS;
    meta.refresh_by = false;
    solver.block_bm_orthonormalize_eig(Y, AY, BY, meta);

    REQUIRE(Y.cols() > 0);
    REQUIRE(Y.cols() <= 3);
    REQUIRE(AY.rows() == Y.rows());
    REQUIRE(AY.cols() == Y.cols());
    REQUIRE(BY.rows() == Y.rows());
    REQUIRE(BY.cols() == Y.cols());

    Matrix Gram      = Y.adjoint() * BY;
    Matrix Gram_symm = (Gram + Gram.adjoint()) * 0.5;
    Matrix I         = Matrix::Identity(Gram.rows(), Gram.cols());
    REQUIRE((Gram_symm - I).norm() < 1e-11);
}

TEST_CASE("generalized gdplusk handles nos4 restart block search") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = grit_test::nos4_matrix<double>();
    Matrix B_matrix = Matrix::Identity(A_matrix.rows(), A_matrix.cols());

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    grit::generalized::gdplusk<double> solver(A, B);
    solver.config.nev                      = 2;
    solver.config.ncv                      = 20;
    solver.config.block_size               = 2;
    solver.config.ritz                     = grit::Ritz::SR;
    solver.config.max_iters                = 200;
    solver.config.abstol                   = 1e-9;
    solver.config.use_b_inner_product      = false;
    solver.config.residual_correction_type = grit::ResidualCorrectionType::CHEAP_OLSEN;
    solver.set_initial_guess(grit_test::seeded_initial_guess<double>(A_matrix.rows(), solver.config.block_size, 31));
    solver.run();

    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> exact(A_matrix, B_matrix);
    auto                                             expected = grit_test::expected_ritz_values(exact.eigenvalues(), solver.config.ritz, solver.config.nev);
    auto                                             view     = solver.get_result();
    REQUIRE_FALSE(grit::has_flag(view.stopReason(), grit::StopReason::invalid_input));
    REQUIRE(std::abs(exact.eigenvalues()(0) - grit_test::nos4_min_eigenvalue) < 1e-12);
    REQUIRE(std::abs(exact.eigenvalues()(exact.eigenvalues().size() - 1) - grit_test::nos4_max_eigenvalue) < 1e-12);
    REQUIRE(std::abs(exact.eigenvalues()(exact.eigenvalues().size() - 1) / exact.eigenvalues()(0) - grit_test::nos4_condition) < 1e-8);
    print_eigenvalue_comparison("generalized gdplusk nos4 restart", view.eigVal(), expected, view.eigVal().size());
    require_close(view.eigVal(), expected, 1e-7);
}

TEST_CASE("generalized gdplusk supports all Ritz targets on nos4") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = grit_test::nos4_matrix<double>();
    Matrix B_matrix = Matrix::Identity(A_matrix.rows(), A_matrix.cols());

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> exact(A_matrix, B_matrix);
    for(auto ritz : {grit::Ritz::SR, grit::Ritz::LR, grit::Ritz::SM, grit::Ritz::LM}) {
        grit::generalized::gdplusk<double> solver(A, B);
        solver.config.nev                      = 2;
        solver.config.ncv                      = A_matrix.rows();
        solver.config.block_size               = 2;
        solver.config.ritz                     = ritz;
        solver.config.max_iters                = 250;
        solver.config.abstol                   = 1e-9;
        solver.config.use_b_inner_product      = true;
        solver.config.residual_correction_type = grit::ResidualCorrectionType::CHEAP_OLSEN;
        solver.set_initial_guess(grit_test::seeded_initial_guess<double>(A_matrix.rows(), solver.config.block_size, 40 + static_cast<int>(ritz)));
        solver.run();

        auto expected = grit_test::expected_ritz_values(exact.eigenvalues(), ritz, solver.config.nev);
        auto view     = solver.get_result();
        REQUIRE_FALSE(grit::has_flag(view.stopReason(), grit::StopReason::invalid_input));
        print_eigenvalue_comparison(std::format("generalized gdplusk {}", grit::enum2sv(ritz)), view.eigVal(), expected, view.eigVal().size());
        require_close(view.eigVal(), expected, 1e-7);
    }
}

TEST_CASE("generalized jacobi-davidson b-only correction supports l2 and bm projectors") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.0, 0.0, 0.5, 2.0, 0.25, 0.0, 0.0, 0.0, 0.25, 5.0, 0.5, 0.0, 0.0, 0.0, 0.5, 6.0;

    Matrix B_matrix = Matrix::Identity(5, 5);
    B_matrix.diagonal() << 1.0, 1.5, 2.0, 2.5, 3.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    Matrix V = grit_test::seeded_initial_guess<double>(A_matrix.rows(), 4, 55);

    grit::generalized::gdplusk<double>::Config cfg;
    cfg.nev             = 1;
    cfg.ncv             = V.cols();
    cfg.block_size      = 1;
    cfg.ritz            = grit::Ritz::SR;
    cfg.max_iters       = 100;
    cfg.inner_max_iters = 20;
    cfg.inner_tol       = 1e-8;
    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> exact(A_matrix, B_matrix);

    SECTION("l2 projectors") {
        auto section_cfg                     = cfg;
        section_cfg.residual_correction_type = grit::ResidualCorrectionType::JACOBI_DAVIDSON;
        section_cfg.use_jd_b_only            = true;
        section_cfg.use_b_inner_product      = false;
        grit::generalized::gdplusk<double> solver(A, B);
        solver.config = section_cfg;
        solver.set_initial_guess(V);

        REQUIRE_NOTHROW(solver.run());
        auto view = solver.get_result();
        print_eigenvalue_comparison("generalized gdplusk jd b-only l2", view.eigVal(), exact.eigenvalues(), view.eigVal().size());
        REQUIRE(view.stopReason() == grit::StopReason::converged);
        REQUIRE(view.num_inner_iters() > 0);
        REQUIRE(view.num_operator_inner() > 0);
        REQUIRE(view.num_matvecs_a_inner() == 0);
        REQUIRE(view.num_matvecs_b_inner() > 0);
        REQUIRE(view.time_solve_inner() > 0.0);
        REQUIRE(view.time_solve_inner() <= view.time());
        REQUIRE(view.time_matvecs_inner() == Approx(view.time_matvecs_a_inner() + view.time_matvecs_b_inner()));
        require_close(view.eigVal(), exact.eigenvalues().head(1), 1e-10);
    }

    SECTION("bm projectors") {
        auto section_cfg                     = cfg;
        section_cfg.residual_correction_type = grit::ResidualCorrectionType::JACOBI_DAVIDSON;
        section_cfg.use_jd_b_only            = true;
        section_cfg.use_b_inner_product      = true;
        grit::generalized::gdplusk<double> solver(A, B);
        solver.config = section_cfg;
        solver.set_initial_guess(V);

        REQUIRE_NOTHROW(solver.run());
        auto view = solver.get_result();
        print_eigenvalue_comparison("generalized gdplusk jd b-only bm", view.eigVal(), exact.eigenvalues(), view.eigVal().size());
        REQUIRE_FALSE(grit::has_flag(view.stopReason(), grit::StopReason::invalid_input));
        REQUIRE(view.eigVal().allFinite());
        REQUIRE(view.rNormsAbs().allFinite());
        REQUIRE(view.num_inner_iters() > 0);
        REQUIRE(view.num_operator_inner() > 0);
        REQUIRE(view.num_matvecs_a_inner() == 0);
        REQUIRE(view.num_matvecs_b_inner() > 0);
        REQUIRE(view.time_solve_inner() > 0.0);
        REQUIRE(view.time_solve_inner() <= view.time());
        require_close(view.eigVal(), exact.eigenvalues().head(1), 1e-10);
    }
}

TEST_CASE("generalized jacobi-davidson correction invokes preconditioner callbacks") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = grit::Matvec<double>::VectorType;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.0, 0.0, 0.5, 2.0, 0.25, 0.0, 0.0, 0.0, 0.25, 5.0, 0.5, 0.0, 0.0, 0.0, 0.5, 6.0;

    Matrix B_matrix = Matrix::Identity(5, 5);
    B_matrix.diagonal() << 1.0, 1.5, 2.0, 2.5, 3.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    int                 update_count = 0;
    int                 apply_count  = 0;
    std::vector<double> theta_values;
    A.set_preconditioner_update([&](double theta) {
        update_count++;
        theta_values.push_back(theta);
    });
    A.set_preconditioner_apply([&](const Eigen::Ref<const Vector> &x, Eigen::Ref<Vector> y, double theta) {
        apply_count++;
        REQUIRE(x.size() == A_matrix.rows());
        REQUIRE(y.size() == A_matrix.rows());
        REQUIRE(std::isfinite(theta));
        y = x;
    });

    grit::generalized::gdplusk<double> solver(A, B);
    solver.config.nev                      = 1;
    solver.config.ncv                      = 4;
    solver.config.block_size               = 1;
    solver.config.ritz                     = grit::Ritz::SR;
    solver.config.max_iters                = 100;
    solver.config.inner_max_iters          = 20;
    solver.config.inner_tol                = 1e-8;
    solver.config.use_b_inner_product      = true;
    solver.config.residual_correction_type = grit::ResidualCorrectionType::JACOBI_DAVIDSON;
    solver.set_initial_guess(grit_test::seeded_initial_guess<double>(A_matrix.rows(), solver.config.ncv, 62));
    solver.run();

    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> exact(A_matrix, B_matrix);
    auto                                             view = solver.get_result();
    print_eigenvalue_comparison("generalized gdplusk jacobi-davidson preconditioner", view.eigVal(), exact.eigenvalues(), view.eigVal().size());
    REQUIRE(view.stopReason() == grit::StopReason::converged);
    REQUIRE(view.num_inner_iters() > 0);
    REQUIRE(view.num_operator_inner() > 0);
    REQUIRE(update_count > 0);
    REQUIRE(apply_count > 0);
    REQUIRE(view.num_preconditioner_updates_total() == update_count);
    REQUIRE(view.num_preconditioner_apply_inner_total() == apply_count);
    REQUIRE(view.time_preconditioner_update_inner() >= 0.0);
    REQUIRE(view.time_preconditioner_apply_inner() >= 0.0);
    REQUIRE(view.time_preconditioner_inner() >= view.time_preconditioner_apply_inner());
    REQUIRE_FALSE(theta_values.empty());
    require_close(view.eigVal(), exact.eigenvalues().head(1), 1e-10);
}

TEST_CASE("generalized jacobi-davidson correction defaults to identity preconditioner") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.0, 0.0, 0.5, 2.0, 0.25, 0.0, 0.0, 0.0, 0.25, 5.0, 0.5, 0.0, 0.0, 0.0, 0.5, 6.0;

    Matrix B_matrix = Matrix::Identity(5, 5);
    B_matrix.diagonal() << 1.0, 1.5, 2.0, 2.5, 3.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    grit::generalized::gdplusk<double> solver(A, B);
    solver.config.nev                      = 1;
    solver.config.ncv                      = 4;
    solver.config.block_size               = 1;
    solver.config.ritz                     = grit::Ritz::SR;
    solver.config.max_iters                = 100;
    solver.config.inner_max_iters          = 20;
    solver.config.inner_tol                = 1e-8;
    solver.config.use_b_inner_product      = false;
    solver.config.residual_correction_type = grit::ResidualCorrectionType::JACOBI_DAVIDSON;
    solver.set_initial_guess(grit_test::seeded_initial_guess<double>(A_matrix.rows(), solver.config.ncv, 63));

    REQUIRE_NOTHROW(solver.run());
    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> exact(A_matrix, B_matrix);
    auto                                             view = solver.get_result();
    print_eigenvalue_comparison("generalized gdplusk identity preconditioner", view.eigVal(), exact.eigenvalues(), view.eigVal().size());
    REQUIRE(view.stopReason() == grit::StopReason::converged);
    REQUIRE(view.num_inner_iters() > 0);
    REQUIRE(view.num_operator_inner() > 0);
    REQUIRE(view.num_matvecs_a_inner() > 0);
    REQUIRE(view.num_matvecs_b_inner() > 0);
    REQUIRE(view.num_precond_total() == 0);
    REQUIRE(view.num_preconditioner_updates_total() == 0);
    REQUIRE(view.num_preconditioner_apply_inner_total() == 0);
    require_close(view.eigVal(), exact.eigenvalues().head(1), 1e-10);
}

TEST_CASE("generalized gdplusk handles small ncv restart without invalid input") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = grit_test::nos4_matrix<double>();
    Matrix B_matrix = Matrix::Identity(A_matrix.rows(), A_matrix.cols());

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    grit::generalized::gdplusk<double> solver(A, B);
    solver.config.nev                      = 1;
    solver.config.ncv                      = 4;
    solver.config.block_size               = 1;
    solver.config.ritz                     = grit::Ritz::SR;
    solver.config.max_iters                = 200;
    solver.config.abstol                   = 1e-8;
    solver.config.use_b_inner_product      = true;
    solver.config.residual_correction_type = grit::ResidualCorrectionType::CHEAP_OLSEN;
    solver.set_initial_guess(grit_test::seeded_initial_guess<double>(A_matrix.rows(), solver.config.ncv, 70));
    solver.run();

    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> exact(A_matrix, B_matrix);
    auto                                             view = solver.get_result();
    print_eigenvalue_comparison("generalized gdplusk small ncv restart", view.eigVal(), exact.eigenvalues(), view.eigVal().size());
    REQUIRE_FALSE(grit::has_flag(view.stopReason(), grit::StopReason::invalid_input));
    REQUIRE(view.stopReason() == grit::StopReason::converged);
    require_close(view.eigVal(), exact.eigenvalues().head(1), 1e-7);
}

TEST_CASE("generalized gdplusk with B as A squared targets A smallest magnitude through LM") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Vector<double, Eigen::Dynamic>;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.0, 0.0, 0.5, 2.0, 0.25, 0.0, 0.0, 0.0, 0.25, 5.0, 0.5, 0.0, 0.0, 0.0, 0.5, 6.0;

    Matrix B_matrix = A_matrix * A_matrix;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    Matrix V = Matrix::Identity(A_matrix.rows(), A_matrix.rows());

    grit::generalized::gdplusk<double> solver(A, B);
    solver.config.nev        = 1;
    solver.config.ncv        = A_matrix.rows();
    solver.config.block_size = 1;
    solver.config.ritz       = grit::Ritz::LM;
    solver.config.max_iters  = 20;
    solver.set_initial_guess(V);
    solver.run();

    Eigen::SelfAdjointEigenSolver<Matrix> exact_A(A_matrix);
    Vector                                expected(1);
    expected << 1.0 / exact_A.eigenvalues()(0);

    auto view = solver.get_result();
    REQUIRE(view.stopReason() == grit::StopReason::converged);
    print_eigenvalue_comparison("generalized gdplusk B=A^2 LM", view.eigVal(), expected, view.eigVal().size());
    write_test_log(std::format("  recovered A eigenvalue {:.16e} exact SM {:.16e} abs_diff {:.3e}\n", 1.0 / view.eigVal()(0), exact_A.eigenvalues()(0),
                               std::abs(1.0 / view.eigVal()(0) - exact_A.eigenvalues()(0))));
    require_close(view.eigVal(), expected, 1e-10);
    REQUIRE(std::abs(1.0 / view.eigVal()(0) - exact_A.eigenvalues()(0)) < 1e-10);
}

TEST_CASE("generalized auto Ritz progress compares the Bv perturbation with the residual") {
    using Matrix     = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorReal = grit::form::base<double>::VectorReal;
    using Correction = grit::ResidualCorrectionType;

    Matrix A_matrix = 2.0 * Matrix::Identity(4, 4);
    Matrix B_matrix = 10.0 * Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto   B        = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    grit::generalized::gdplusk<double> solver(A, B);
    solver.config.nev                                 = 1;
    solver.config.ncv                                 = 4;
    solver.config.block_size                          = 1;
    solver.config.abstol                              = 1e-6;
    solver.config.residual_correction_type            = Correction::AUTO;
    solver.V                                          = Matrix::Identity(4, 1);
    solver.BV                                         = B_matrix * solver.V;
    solver.status.eigVal                              = VectorReal::Constant(1, 0.2);
    solver.status.rNormsAbs                           = VectorReal::Ones(1);
    solver.status.eigVals_history                     = {VectorReal::Constant(1, 0.198), VectorReal::Constant(1, 0.199), VectorReal::Constant(1, 0.2),
                                                         VectorReal::Constant(1, 0.201), VectorReal::Constant(1, 0.202)};
    solver.auto_residual_correction.cheap_olsen_iters = 4;

    SECTION("motion above the residual-scaled tolerance remains in cheap Olsen mode") {
        solver.config.ritz_stabilization_tolerance = 9e-3;
        solver.update_auto_residual_correction_state();
        REQUIRE(solver.auto_residual_correction.active == Correction::CHEAP_OLSEN);
    }

    SECTION("motion below the residual-scaled tolerance starts JD") {
        solver.config.ritz_stabilization_tolerance = 11e-3;
        solver.update_auto_residual_correction_state();
        REQUIRE(solver.auto_residual_correction.active == Correction::JACOBI_DAVIDSON);
    }
}

int main(int argc, char **argv) {
    test_log_fd = ::dup(STDERR_FILENO);
    Catch::Session session;
    const int      return_code = session.applyCommandLine(argc, argv);
    if(return_code != 0) return return_code;
    const int result = session.run();
    if(test_log_fd >= 0) ::close(test_log_fd);
    return result;
}

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include "solver_test_utils.h"
#include <cmath>
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
    void require_close(const VecA &a, const VecB &b, double tol) {
        REQUIRE(a.size() == b.size());
        for(Eigen::Index i = 0; i < a.size(); ++i) REQUIRE(std::abs(a(i) - b(i)) < tol);
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

TEST_CASE("standard gdplusk matches dense eigensolver") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.0, 0.0, 0.5, 2.0, 0.25, 0.0, 0.0, 0.0, 0.25, 5.0, 0.5, 0.0, 0.0, 0.0, 0.5, 6.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    auto solver              = grit::standard::gdplusk<double>(A);
    solver.config.nev        = 1;
    solver.config.ncv        = A_matrix.rows();
    solver.config.block_size = 1;
    solver.config.ritz       = grit::OptRitz::SR;
    solver.config.max_iters  = 20;
    solver.set_initial_guess(Matrix::Identity(A_matrix.rows(), A_matrix.rows()));
    solver.run();

    Eigen::SelfAdjointEigenSolver<Matrix> exact(A_matrix);
    auto                                  view = grit::solver_view<double>(solver);
    REQUIRE(view.stopReason() == grit::StopReason::converged);
    print_eigenvalue_comparison("standard gdplusk", view.eigVal(), exact.eigenvalues(), view.eigVal().size());
    require_close(view.eigVal(), exact.eigenvalues().head(1), 1e-10);
}

TEST_CASE("standard gdplusk handles nos4 restart block search") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = grit_test::nos4_matrix<double>();
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev                      = 2;
    solver.config.ncv                      = 20;
    solver.config.block_size               = 2;
    solver.config.ritz                     = grit::OptRitz::SR;
    solver.config.max_iters                = 200;
    solver.config.tol                      = 1e-9;
    solver.config.residual_correction_type = grit::ResidualCorrectionType::CHEAP_OLSEN;
    solver.set_initial_guess(grit_test::seeded_initial_guess<double>(A_matrix.rows(), solver.config.block_size, 11));
    solver.run();

    Eigen::SelfAdjointEigenSolver<Matrix> exact(A_matrix);
    auto                                  expected = grit_test::expected_ritz_values(exact.eigenvalues(), solver.config.ritz, solver.config.nev);
    auto                                  view     = grit::solver_view<double>(solver);
    REQUIRE_FALSE(grit::has_flag(view.stopReason(), grit::StopReason::invalid_input));
    REQUIRE(std::abs(exact.eigenvalues()(0) - grit_test::nos4_min_eigenvalue) < 1e-12);
    REQUIRE(std::abs(exact.eigenvalues()(exact.eigenvalues().size() - 1) - grit_test::nos4_max_eigenvalue) < 1e-12);
    REQUIRE(std::abs(exact.eigenvalues()(exact.eigenvalues().size() - 1) / exact.eigenvalues()(0) - grit_test::nos4_condition) < 1e-8);
    print_eigenvalue_comparison("standard gdplusk nos4 restart", view.eigVal(), expected, view.eigVal().size());
    require_close(view.eigVal(), expected, 1e-7);
}

TEST_CASE("standard gdplusk supports all Ritz targets on nos4") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = grit_test::nos4_matrix<double>();
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    Eigen::SelfAdjointEigenSolver<Matrix> exact(A_matrix);
    for(auto ritz : {grit::OptRitz::SR, grit::OptRitz::LR, grit::OptRitz::SM, grit::OptRitz::LM}) {
        grit::standard::gdplusk<double> solver(A);
        solver.config.nev                      = 2;
        solver.config.ncv                      = A_matrix.rows();
        solver.config.block_size               = 2;
        solver.config.ritz                     = ritz;
        solver.config.max_iters                = 250;
        solver.config.tol                      = 1e-9;
        solver.config.residual_correction_type = grit::ResidualCorrectionType::CHEAP_OLSEN;
        solver.set_initial_guess(grit_test::seeded_initial_guess<double>(A_matrix.rows(), solver.config.block_size, 20 + static_cast<int>(ritz)));
        solver.run();

        auto expected = grit_test::expected_ritz_values(exact.eigenvalues(), ritz, solver.config.nev);
        auto view     = grit::solver_view<double>(solver);
        REQUIRE_FALSE(grit::has_flag(view.stopReason(), grit::StopReason::invalid_input));
        print_eigenvalue_comparison(std::format("standard gdplusk {}", grit::enum2sv(ritz)), view.eigVal(), expected, view.eigVal().size());
        require_close(view.eigVal(), expected, 1e-7);
    }
}

TEST_CASE("standard gdplusk converges with an exact zero eigenvalue") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(4, 4);
    A_matrix << 1.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, -1.0, 1.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    Matrix V = Matrix::Identity(A_matrix.rows(), A_matrix.rows());

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev        = 1;
    solver.config.ncv        = A_matrix.rows();
    solver.config.block_size = 1;
    solver.config.ritz       = grit::OptRitz::SR;
    solver.config.max_iters  = 20;
    solver.config.tol        = 1e-12;
    solver.set_initial_guess(V);
    solver.run();

    Eigen::SelfAdjointEigenSolver<Matrix> exact(A_matrix);
    auto                                  view = grit::solver_view<double>(solver);
    REQUIRE(view.stopReason() == grit::StopReason::converged);
    print_eigenvalue_comparison("standard gdplusk zero eigenvalue", view.eigVal(), exact.eigenvalues(), view.eigVal().size());
    REQUIRE(std::abs(view.eigVal()(0)) < 1e-12);
    REQUIRE(view.rNorms()(0) < 1e-12);
}

TEST_CASE("standard gdplusk user callback reports initial and final view") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.0, 0.0, 0.5, 2.0, 0.25, 0.0, 0.0, 0.0, 0.25, 5.0, 0.5, 0.0, 0.0, 0.0, 0.5, 6.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    Matrix V = Matrix::Identity(A_matrix.rows(), A_matrix.rows());

    std::vector<Eigen::Index>       iterations;
    std::vector<grit::StopReason>   stop_reasons;
    grit::standard::gdplusk<double> solver(A);
    solver.config.nev           = 1;
    solver.config.ncv           = A_matrix.rows();
    solver.config.block_size    = 1;
    solver.config.ritz          = grit::OptRitz::SR;
    solver.config.max_iters     = 20;
    solver.config.user_callback = [&](const auto &solver_ref) {
        auto view = grit::solver_view<double>(solver_ref);
        iterations.push_back(view.iter());
        stop_reasons.push_back(view.stopReason());
    };
    solver.set_initial_guess(V);
    solver.run();

    Eigen::SelfAdjointEigenSolver<Matrix> exact(A_matrix);
    auto                                  view = grit::solver_view<double>(solver);
    print_eigenvalue_comparison("standard gdplusk callback", view.eigVal(), exact.eigenvalues(), view.eigVal().size());

    require_close(view.eigVal(), exact.eigenvalues().head(1), 1e-10);
    REQUIRE(iterations.size() >= 2);
    REQUIRE(iterations.front() == 0);
    REQUIRE(iterations.back() + 1 == view.iter());
    REQUIRE(stop_reasons.front() == grit::StopReason::none);
    REQUIRE(stop_reasons.back() == grit::StopReason::converged);
}

TEST_CASE("standard jacobi-davidson correction invokes preconditioner callbacks") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = grit::Matvec<double>::VectorType;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.0, 0.0, 0.5, 2.0, 0.25, 0.0, 0.0, 0.0, 0.25, 5.0, 0.5, 0.0, 0.0, 0.0, 0.5, 6.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    int update_count = 0;
    int apply_count  = 0;
    A.set_preconditioner_update([&](double) { update_count++; });
    A.set_preconditioner_apply([&](const Eigen::Ref<const Vector> &x, Eigen::Ref<Vector> y, double) {
        apply_count++;
        y = x;
    });

    Matrix V(A_matrix.rows(), 3);
    V << 1.0, 0.2, 0.3, 0.4, 1.0, 0.2, 0.3, 0.5, 1.0, 0.2, 0.4, 0.5, 0.5, 0.3, 0.4;

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev                      = 1;
    solver.config.ncv                      = 3;
    solver.config.block_size               = 1;
    solver.config.ritz                     = grit::OptRitz::SR;
    solver.config.max_iters                = 100;
    solver.config.inner_max_iters          = 20;
    solver.config.inner_tol                = 1e-8;
    solver.config.residual_correction_type = grit::ResidualCorrectionType::JACOBI_DAVIDSON;
    solver.set_initial_guess(V);
    solver.run();

    Eigen::SelfAdjointEigenSolver<Matrix> exact(A_matrix);
    auto                                  view = grit::solver_view<double>(solver);
    print_eigenvalue_comparison("standard gdplusk jacobi-davidson", view.eigVal(), exact.eigenvalues(), view.eigVal().size());

    REQUIRE(view.stopReason() == grit::StopReason::converged);
    require_close(view.eigVal(), exact.eigenvalues().head(1), 1e-10);
    REQUIRE(update_count > 0);
    REQUIRE(apply_count > 0);
}

TEST_CASE("standard jacobi-davidson correction defaults to identity preconditioner") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.0, 0.0, 0.5, 2.0, 0.25, 0.0, 0.0, 0.0, 0.25, 5.0, 0.5, 0.0, 0.0, 0.0, 0.5, 6.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    Matrix V(A_matrix.rows(), 3);
    V << 1.0, 0.2, 0.3, 0.4, 1.0, 0.2, 0.3, 0.5, 1.0, 0.2, 0.4, 0.5, 0.5, 0.3, 0.4;

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev                      = 1;
    solver.config.ncv                      = 3;
    solver.config.block_size               = 1;
    solver.config.ritz                     = grit::OptRitz::SR;
    solver.config.max_iters                = 100;
    solver.config.inner_max_iters          = 20;
    solver.config.inner_tol                = 1e-8;
    solver.config.residual_correction_type = grit::ResidualCorrectionType::JACOBI_DAVIDSON;
    solver.set_initial_guess(V);

    REQUIRE_NOTHROW(solver.run());
    Eigen::SelfAdjointEigenSolver<Matrix> exact(A_matrix);
    auto                                  view = grit::solver_view<double>(solver);
    print_eigenvalue_comparison("standard gdplusk identity preconditioner", view.eigVal(), exact.eigenvalues(), view.eigVal().size());
    REQUIRE(view.stopReason() == grit::StopReason::converged);
    require_close(view.eigVal(), exact.eigenvalues().head(1), 1e-10);
}

TEST_CASE("standard auto residual correction starts with cheap Olsen") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev        = 1;
    solver.config.ncv        = 4;
    solver.config.block_size = 1;
    using Correction         = grit::form::base<double>::ResidualCorrectionType;

    solver.config.residual_correction_type = Correction::AUTO;
    solver.adjust_residual_correction_type();

    REQUIRE(solver.residual_correction_type_internal == Correction::CHEAP_OLSEN);
    REQUIRE(solver.auto_residual_correction.active == Correction::CHEAP_OLSEN);
    REQUIRE(solver.auto_residual_correction.step_method == Correction::CHEAP_OLSEN);
}

TEST_CASE("standard auto residual correction does not start Jacobi-Davidson unless eigenvalues and residuals saturate") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev        = 1;
    solver.config.ncv        = 4;
    solver.config.block_size = 1;
    using Base               = grit::form::base<double>;
    using Correction         = Base::ResidualCorrectionType;
    using VectorReal         = Base::VectorReal;

    solver.config.residual_correction_type      = Correction::AUTO;
    solver.auto_residual_correction.active      = Correction::CHEAP_OLSEN;
    solver.auto_residual_correction.step_method = Correction::CHEAP_OLSEN;
    solver.auto_residual_correction.dwell       = solver.config.auto_min_dwell_iters;
    solver.status.iter                          = 2;
    solver.status.eigVal                        = VectorReal::Constant(1, -1.0);
    solver.status.rNorms                        = VectorReal::Constant(1, 1.0e-2);
    solver.status.eigVals_history.clear();
    solver.status.rNorms_history.clear();
    solver.status.eigVals_history.emplace_back(VectorReal::Constant(1, -1.0));
    solver.status.eigVals_history.emplace_back(VectorReal::Constant(1, -1.5));
    solver.status.rNorms_history.emplace_back(VectorReal::Constant(1, 1.0e-2));
    solver.status.rNorms_history.emplace_back(VectorReal::Constant(1, 1.0e-2 + 1.0e-8));
    solver.status.num_matvecs       = 1;
    solver.status.num_matvecs_inner = 0;

    solver.update_auto_residual_correction_state();

    REQUIRE(solver.auto_residual_correction.active == Correction::CHEAP_OLSEN);
    REQUIRE(solver.auto_residual_correction.cheap_to_jd_switch_iters.empty());
}

TEST_CASE("standard auto eigenvalue saturation is relative to average eigenvalue magnitude") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev                  = 1;
    solver.config.ncv                  = 4;
    solver.config.block_size           = 1;
    solver.config.sat_eigval_threshold = 1e-3;

    using VectorReal     = grit::form::base<double>::VectorReal;
    solver.status.iter   = 2;
    solver.status.eigVal = VectorReal::Constant(1, 1.0e6);
    solver.status.eigVals_history.clear();
    solver.status.eigVals_history.emplace_back(VectorReal::Constant(1, 1.0e6));
    solver.status.eigVals_history.emplace_back(VectorReal::Constant(1, 1.0e6 + 5.0e2));

    REQUIRE(solver.eigVals_have_saturated());
}

int main(int argc, char **argv) {
    test_log_fd = ::dup(STDERR_FILENO);
    Catch::Session session;
    const int      return_code = session.applyCommandLine(argc, argv);
    if(return_code != 0) return return_code;
    session.configData().showSuccessfulTests = true;
    session.configData().reporterName        = "compact";
    const int result                         = session.run();
    if(test_log_fd >= 0) ::close(test_log_fd);
    return result;
}

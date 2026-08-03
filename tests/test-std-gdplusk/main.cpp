#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include "solver_test_utils.h"
#include <cmath>
#include <Eigen/Eigenvalues>
#include <format>
#include <grit/grit.h>
#include <spdlog/sinks/ostream_sink.h>
#include <sstream>
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

TEST_CASE("standard gdplusk matches dense eigensolver") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.0, 0.0, 0.5, 2.0, 0.25, 0.0, 0.0, 0.0, 0.25, 5.0, 0.5, 0.0, 0.0, 0.0, 0.5, 6.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    auto solver              = grit::standard::gdplusk<double>(A);
    solver.config.nev        = 1;
    solver.config.ncv        = A_matrix.rows();
    solver.config.block_size = 1;
    solver.config.ritz       = grit::Ritz::SR;
    solver.config.max_iters  = 20;
    solver.set_initial_guess(Matrix::Identity(A_matrix.rows(), A_matrix.rows()));
    solver.run();

    Eigen::SelfAdjointEigenSolver<Matrix> exact(A_matrix);
    auto                                  view = solver.get_result();
    REQUIRE(view.stopReason() == grit::StopReason::converged);
    print_eigenvalue_comparison("standard gdplusk", view.eigVal(), exact.eigenvalues(), view.eigVal().size());
    require_close(view.eigVal(), exact.eigenvalues().head(1), 1e-10);
}

TEST_CASE("standard gdplusk owns temporary initial guess") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.0, 0.0, 0.5, 2.0, 0.25, 0.0, 0.0, 0.0, 0.25, 5.0, 0.5, 0.0, 0.0, 0.0, 0.5, 6.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev                      = 1;
    solver.config.ncv                      = 3;
    solver.config.block_size               = 1;
    solver.config.ritz                     = grit::Ritz::SR;
    solver.config.max_iters                = 100;
    solver.config.abstol                   = 1e-12;
    solver.config.residual_correction_type = grit::ResidualCorrectionType::CHEAP_OLSEN;
    solver.set_initial_guess(grit_test::seeded_initial_guess<double>(A_matrix.rows(), solver.config.ncv, 10));
    solver.run();

    Eigen::SelfAdjointEigenSolver<Matrix> exact(A_matrix);
    auto                                  view = solver.get_result();
    print_eigenvalue_comparison("standard gdplusk temporary initial guess", view.eigVal(), exact.eigenvalues(), view.eigVal().size());
    REQUIRE(view.stopReason() == grit::StopReason::converged);
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
    solver.config.ritz                     = grit::Ritz::SR;
    solver.config.max_iters                = 200;
    solver.config.abstol                   = 1e-9;
    solver.config.residual_correction_type = grit::ResidualCorrectionType::CHEAP_OLSEN;
    solver.set_initial_guess(grit_test::seeded_initial_guess<double>(A_matrix.rows(), solver.config.block_size, 11));
    solver.run();

    Eigen::SelfAdjointEigenSolver<Matrix> exact(A_matrix);
    auto                                  expected = grit_test::expected_ritz_values(exact.eigenvalues(), solver.config.ritz, solver.config.nev);
    auto                                  view     = solver.get_result();
    REQUIRE_FALSE(grit::has_flag(view.stopReason(), grit::StopReason::invalid_input));
    REQUIRE(std::abs(exact.eigenvalues()(0) - grit_test::nos4_min_eigenvalue) < 1e-12);
    REQUIRE(std::abs(exact.eigenvalues()(exact.eigenvalues().size() - 1) - grit_test::nos4_max_eigenvalue) < 1e-12);
    REQUIRE(std::abs(exact.eigenvalues()(exact.eigenvalues().size() - 1) / exact.eigenvalues()(0) - grit_test::nos4_condition) < 1e-8);
    print_eigenvalue_comparison("standard gdplusk nos4 restart", view.eigVal(), expected, view.eigVal().size());
    require_close(view.eigVal(), expected, 1e-7);
}

TEST_CASE("standard gdplusk handles small ncv restart without invalid input") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = grit_test::nos4_matrix<double>();
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev                      = 1;
    solver.config.ncv                      = 4;
    solver.config.block_size               = 1;
    solver.config.ritz                     = grit::Ritz::SR;
    solver.config.max_iters                = 200;
    solver.config.abstol                   = 1e-8;
    solver.config.residual_correction_type = grit::ResidualCorrectionType::CHEAP_OLSEN;
    solver.set_initial_guess(grit_test::seeded_initial_guess<double>(A_matrix.rows(), solver.config.ncv, 12));
    solver.run();

    Eigen::SelfAdjointEigenSolver<Matrix> exact(A_matrix);
    auto                                  view = solver.get_result();
    print_eigenvalue_comparison("standard gdplusk small ncv restart", view.eigVal(), exact.eigenvalues(), view.eigVal().size());
    REQUIRE_FALSE(grit::has_flag(view.stopReason(), grit::StopReason::invalid_input));
    REQUIRE(view.eigVal().allFinite());
    REQUIRE(view.rNormsAbs().allFinite());
}

TEST_CASE("standard gdplusk supports all Ritz targets on nos4") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = grit_test::nos4_matrix<double>();
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    Eigen::SelfAdjointEigenSolver<Matrix> exact(A_matrix);
    for(auto ritz : {grit::Ritz::SR, grit::Ritz::LR, grit::Ritz::SM, grit::Ritz::LM}) {
        grit::standard::gdplusk<double> solver(A);
        solver.config.nev                      = 2;
        solver.config.ncv                      = A_matrix.rows();
        solver.config.block_size               = 2;
        solver.config.ritz                     = ritz;
        solver.config.max_iters                = 250;
        solver.config.abstol                   = 1e-9;
        solver.config.residual_correction_type = grit::ResidualCorrectionType::CHEAP_OLSEN;
        solver.set_initial_guess(grit_test::seeded_initial_guess<double>(A_matrix.rows(), solver.config.block_size, 20 + static_cast<int>(ritz)));
        solver.run();

        auto expected = grit_test::expected_ritz_values(exact.eigenvalues(), ritz, solver.config.nev);
        auto view     = solver.get_result();
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
    solver.config.ritz       = grit::Ritz::SR;
    solver.config.max_iters  = 20;
    solver.config.abstol     = 1e-12;
    solver.set_initial_guess(V);
    solver.run();

    Eigen::SelfAdjointEigenSolver<Matrix> exact(A_matrix);
    auto                                  view = solver.get_result();
    REQUIRE(view.stopReason() == grit::StopReason::converged);
    print_eigenvalue_comparison("standard gdplusk zero eigenvalue", view.eigVal(), exact.eigenvalues(), view.eigVal().size());
    REQUIRE(std::abs(view.eigVal()(0)) < 1e-12);
    REQUIRE(view.rNormsAbs()(0) < 1e-12);
}

TEST_CASE("standard gdplusk user callback reports initial and final view") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.0, 0.0, 0.5, 2.0, 0.25, 0.0, 0.0, 0.0, 0.25, 5.0, 0.5, 0.0, 0.0, 0.0, 0.5, 6.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    Matrix V = Matrix::Identity(A_matrix.rows(), A_matrix.rows());

    std::vector<Eigen::Index>       outer_iterations;
    std::vector<grit::StopReason>   stop_reasons;
    grit::standard::gdplusk<double> solver(A);
    solver.config.nev           = 1;
    solver.config.ncv           = A_matrix.rows();
    solver.config.block_size    = 1;
    solver.config.ritz          = grit::Ritz::SR;
    solver.config.max_iters     = 20;
    solver.config.user_callback = [&](const auto &solver_ref) {
        auto view = solver_ref.get_result_view();
        outer_iterations.push_back(view.outer_iter());
        stop_reasons.push_back(view.stopReason());
    };
    solver.set_initial_guess(V);
    solver.run();

    Eigen::SelfAdjointEigenSolver<Matrix> exact(A_matrix);
    auto                                  view = solver.get_result();
    print_eigenvalue_comparison("standard gdplusk callback", view.eigVal(), exact.eigenvalues(), view.eigVal().size());

    require_close(view.eigVal(), exact.eigenvalues().head(1), 1e-10);
    REQUIRE(outer_iterations.size() >= 2);
    REQUIRE(outer_iterations.front() == 0);
    REQUIRE(outer_iterations.back() + 1 == view.outer_iter());
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
    solver.config.ritz                     = grit::Ritz::SR;
    solver.config.max_iters                = 100;
    solver.config.inner_max_iters          = 20;
    solver.config.inner_tol                = 1e-8;
    solver.config.residual_correction_type = grit::ResidualCorrectionType::JACOBI_DAVIDSON;
    solver.set_initial_guess(V);
    solver.run();

    Eigen::SelfAdjointEigenSolver<Matrix> exact(A_matrix);
    auto                                  view = solver.get_result();
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
    solver.config.ritz                     = grit::Ritz::SR;
    solver.config.max_iters                = 100;
    solver.config.inner_max_iters          = 20;
    solver.config.inner_tol                = 1e-8;
    solver.config.residual_correction_type = grit::ResidualCorrectionType::JACOBI_DAVIDSON;
    solver.set_initial_guess(V);

    REQUIRE_NOTHROW(solver.run());
    Eigen::SelfAdjointEigenSolver<Matrix> exact(A_matrix);
    auto                                  view = solver.get_result();
    print_eigenvalue_comparison("standard gdplusk identity preconditioner", view.eigVal(), exact.eigenvalues(), view.eigVal().size());
    REQUIRE(view.stopReason() == grit::StopReason::converged);
    require_close(view.eigVal(), exact.eigenvalues().head(1), 1e-10);
}

TEST_CASE("standard history slopes use the requested trailing entries") {
    using Matrix     = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorReal = grit::form::base<double>::VectorReal;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    std::deque<VectorReal>          history = {
        (VectorReal(2) << 1e3, 8.0).finished(),
        (VectorReal(2) << 1e2, 4.0).finished(),
        (VectorReal(2) << 1e1, 2.0).finished(),
        (VectorReal(2) << 1e0, 1.0).finished(),
    };

    VectorReal slope_errors;
    auto       slopes = solver.get_slopes(history, true, -1, &slope_errors);
    REQUIRE(std::abs(slopes(0) + 1.0) < 1e-12);
    REQUIRE(std::abs(slopes(1) + std::log10(2.0)) < 1e-12);
    REQUIRE(slope_errors.norm() < 1e-12);

    auto negative_last_n = solver.get_slopes(history, true, -7);
    REQUIRE((negative_last_n - slopes).norm() < 1e-12);

    auto last_two = solver.get_slopes(history, false, 2, &slope_errors);
    REQUIRE(std::abs(last_two(0) + 9.0) < 1e-12);
    REQUIRE(std::abs(last_two(1) + 1.0) < 1e-12);
    REQUIRE(slope_errors.array().isNaN().all());

    auto last_ten = solver.get_slopes(history, true, 10);
    REQUIRE((last_ten - slopes).norm() < 1e-12);
    REQUIRE(solver.get_slopes(history, true, 0).array().isNaN().all());
    REQUIRE(solver.get_slopes(history, true, 1).array().isNaN().all());

    history = {VectorReal::Constant(1, 0.0), VectorReal::Constant(1, 0.0), VectorReal::Constant(1, 0.0), VectorReal::Constant(1, 0.0),
               VectorReal::Constant(1, 1.0)};
    slopes  = solver.get_slopes(history, false, -1, &slope_errors);
    REQUIRE(std::abs(slopes(0) - 0.2) < 1e-12);
    REQUIRE(std::abs(slopes(0)) < 2.0 * slope_errors(0));
}

TEST_CASE("standard auto residual correction starts with cheap Olsen and schedules probes from residual slopes") {
    using Matrix     = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorReal = grit::form::base<double>::VectorReal;
    using Correction = grit::ResidualCorrectionType;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev                      = 1;
    solver.config.ncv                      = 4;
    solver.config.block_size               = 1;
    solver.config.residual_correction_type = Correction::AUTO;
    REQUIRE(solver.config.auto_probe_length == 3);
    REQUIRE(solver.config.auto_max_probes == 1);

    solver.adjust_residual_correction_type();
    REQUIRE(solver.residual_correction_type_internal == Correction::CHEAP_OLSEN);

    solver.status.rNormsAbs                                    = VectorReal::Ones(1);
    solver.status.rNormsAbsHistory                             = {VectorReal::Ones(1), VectorReal::Ones(1)};
    solver.auto_residual_correction.active                     = Correction::JACOBI_DAVIDSON;
    solver.auto_residual_correction.jd_outer_iters_since_probe = 2;
    solver.adjust_residual_correction_type();
    REQUIRE(solver.auto_residual_correction.iteration_method == Correction::CHEAP_OLSEN);
}

TEST_CASE("standard auto residual correction probes when an unconverged residual slope is nonnegative") {
    using Matrix     = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorReal = grit::form::base<double>::VectorReal;
    using Correction = grit::ResidualCorrectionType;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev                                          = 2;
    solver.config.abstol                                       = 1e-6;
    solver.config.residual_correction_type                     = Correction::AUTO;
    solver.status.rNormsAbs                                    = VectorReal::Ones(2);
    solver.auto_residual_correction.active                     = Correction::JACOBI_DAVIDSON;
    solver.auto_residual_correction.jd_outer_iters_since_probe = 3;

    SECTION("negative slopes continue JD") {
        solver.status.rNormsAbsHistory = {(VectorReal(2) << 1.0, 1.0).finished(), (VectorReal(2) << 0.5, 0.25).finished(),
                                          (VectorReal(2) << 0.25, 0.125).finished()};
        solver.adjust_residual_correction_type();
        REQUIRE(solver.auto_residual_correction.iteration_method == Correction::JACOBI_DAVIDSON);
    }

    SECTION("a zero slope starts a probe") {
        solver.status.rNormsAbsHistory = {(VectorReal(2) << 1.0, 1.0).finished(), (VectorReal(2) << 0.5, 1.0).finished(),
                                          (VectorReal(2) << 0.25, 1.0).finished()};
        solver.adjust_residual_correction_type();
        REQUIRE(solver.auto_residual_correction.iteration_method == Correction::CHEAP_OLSEN);
    }

    SECTION("a positive slope starts a probe") {
        solver.status.rNormsAbsHistory = {(VectorReal(2) << 1.0, 1.0).finished(), (VectorReal(2) << 0.5, 2.0).finished(),
                                          (VectorReal(2) << 0.25, 4.0).finished()};
        solver.adjust_residual_correction_type();
        REQUIRE(solver.auto_residual_correction.iteration_method == Correction::CHEAP_OLSEN);
    }

    SECTION("a converged slot does not start a probe") {
        solver.status.rNormsAbs        = (VectorReal(2) << 1.0, 1e-8).finished();
        solver.status.rNormsAbsHistory = {(VectorReal(2) << 1.0, 1.0).finished(), (VectorReal(2) << 0.5, 2.0).finished(),
                                          (VectorReal(2) << 0.25, 4.0).finished()};
        solver.adjust_residual_correction_type();
        REQUIRE(solver.auto_residual_correction.iteration_method == Correction::JACOBI_DAVIDSON);
    }

    SECTION("zero or one JD entry cannot start a probe") {
        solver.status.rNormsAbsHistory                             = {VectorReal::Ones(2), VectorReal::Ones(2)};
        solver.auto_residual_correction.jd_outer_iters_since_probe = 1;
        solver.adjust_residual_correction_type();
        REQUIRE(solver.auto_residual_correction.iteration_method == Correction::JACOBI_DAVIDSON);
    }

    SECTION("only the trailing consecutive JD entries are used") {
        solver.status.rNormsAbsHistory                             = {(VectorReal(2) << 1.0, 1.0).finished(), (VectorReal(2) << 2.0, 2.0).finished(),
                                                                      (VectorReal(2) << 4.0, 4.0).finished(), (VectorReal(2) << 2.0, 2.0).finished(),
                                                                      (VectorReal(2) << 1.0, 1.0).finished()};
        solver.auto_residual_correction.jd_outer_iters_since_probe = 2;
        solver.adjust_residual_correction_type();
        REQUIRE(solver.auto_residual_correction.iteration_method == Correction::JACOBI_DAVIDSON);
    }

    SECTION("the configured history size limits the slope window") {
        solver.status.max_history_size                             = 3;
        solver.status.rNormsAbsHistory                             = {(VectorReal(2) << 1e3, 1.0).finished(), (VectorReal(2) << 1e2, 0.5).finished(),
                                                                      (VectorReal(2) << 1.0, 0.25).finished(), (VectorReal(2) << 2.0, 0.125).finished(),
                                                                      (VectorReal(2) << 4.0, 0.0625).finished()};
        solver.auto_residual_correction.jd_outer_iters_since_probe = 5;
        solver.adjust_residual_correction_type();
        REQUIRE(solver.auto_residual_correction.iteration_method == Correction::CHEAP_OLSEN);
    }
}

TEST_CASE("standard auto residual correction limits probes in a stabilized Ritz basin") {
    using Matrix     = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorReal = grit::form::base<double>::VectorReal;
    using Correction = grit::ResidualCorrectionType;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.residual_correction_type                     = Correction::AUTO;
    solver.status.rNormsAbs                                    = VectorReal::Ones(1);
    solver.status.rNormsAbsHistory                             = {VectorReal::Ones(1), VectorReal::Ones(1)};
    solver.auto_residual_correction.active                     = Correction::JACOBI_DAVIDSON;
    solver.auto_residual_correction.jd_outer_iters_since_probe = 2;

    SECTION("zero disables probes") {
        solver.config.auto_max_probes = 0;
        solver.adjust_residual_correction_type();
        REQUIRE(solver.auto_residual_correction.iteration_method == Correction::JACOBI_DAVIDSON);
    }

    SECTION("the default permits one probe") {
        solver.adjust_residual_correction_type();
        REQUIRE(solver.auto_residual_correction.iteration_method == Correction::CHEAP_OLSEN);

        solver.auto_residual_correction.probes_started    = 1;
        solver.auto_residual_correction.cheap_olsen_iters = 1;
        solver.adjust_residual_correction_type();
        REQUIRE(solver.auto_residual_correction.iteration_method == Correction::CHEAP_OLSEN);

        solver.auto_residual_correction.cheap_olsen_iters = 0;
        solver.adjust_residual_correction_type();
        REQUIRE(solver.auto_residual_correction.iteration_method == Correction::JACOBI_DAVIDSON);
    }

    SECTION("positive values permit the requested number") {
        solver.config.auto_max_probes                  = 2;
        solver.auto_residual_correction.probes_started = 1;
        solver.adjust_residual_correction_type();
        REQUIRE(solver.auto_residual_correction.iteration_method == Correction::CHEAP_OLSEN);

        solver.auto_residual_correction.probes_started = 2;
        solver.adjust_residual_correction_type();
        REQUIRE(solver.auto_residual_correction.iteration_method == Correction::JACOBI_DAVIDSON);
    }

    SECTION("minus one permits unlimited probes") {
        solver.config.auto_max_probes                  = -1;
        solver.auto_residual_correction.probes_started = 100;
        solver.adjust_residual_correction_type();
        REQUIRE(solver.auto_residual_correction.iteration_method == Correction::CHEAP_OLSEN);
    }
}

TEST_CASE("standard auto residual correction does not probe near the residual target") {
    using Matrix     = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorReal = grit::form::base<double>::VectorReal;
    using Correction = grit::ResidualCorrectionType;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.abstol                                       = 1e-2;
    solver.config.residual_correction_type                     = Correction::AUTO;
    solver.auto_residual_correction.active                     = Correction::JACOBI_DAVIDSON;
    solver.auto_residual_correction.jd_outer_iters_since_probe = 2;

    SECTION("residual below target divided by inner tolerance stays in JD") {
        solver.status.rNormsAbs        = VectorReal::Constant(1, 9e-2);
        solver.status.rNormsAbsHistory = {solver.status.rNormsAbs, solver.status.rNormsAbs};
        solver.adjust_residual_correction_type();
        REQUIRE(solver.auto_residual_correction.iteration_method == Correction::JACOBI_DAVIDSON);
    }

    SECTION("residual above target divided by inner tolerance permits a probe") {
        solver.status.rNormsAbs        = VectorReal::Constant(1, 1.1e-1);
        solver.status.rNormsAbsHistory = {solver.status.rNormsAbs, solver.status.rNormsAbs};
        solver.adjust_residual_correction_type();
        REQUIRE(solver.auto_residual_correction.iteration_method == Correction::CHEAP_OLSEN);
    }
}

TEST_CASE("standard auto residual correction requires three Ritz entries and ignores very slow drift") {
    using Matrix     = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorReal = grit::form::base<double>::VectorReal;
    using Correction = grit::ResidualCorrectionType;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev                      = 1;
    solver.config.ncv                      = 4;
    solver.config.block_size               = 1;
    solver.config.abstol                   = 1e-6;
    solver.config.residual_correction_type = Correction::AUTO;
    solver.V                               = Matrix::Identity(4, 1);
    solver.status.eigVal                   = VectorReal::Constant(1, 1e-8);
    solver.status.rNormsAbs                = VectorReal::Constant(1, 1.0);
    solver.status.max_history_size         = 9;
    solver.status.rNormsAbsHistory         = {VectorReal::Constant(1, 1e-1), VectorReal::Constant(1, 1e-3), VectorReal::Constant(1, 1e-1),
                                              VectorReal::Constant(1, 1e-3), VectorReal::Constant(1, 1e-1)};
    solver.status.eigVals_history          = {VectorReal::Constant(1, -1e-6), VectorReal::Constant(1, 0.0)};

    solver.update_auto_residual_correction_state();
    REQUIRE(solver.auto_residual_correction.active == Correction::CHEAP_OLSEN);

    solver.status.eigVals_history.emplace_back(VectorReal::Constant(1, 1e-6));
    solver.update_auto_residual_correction_state();
    REQUIRE(solver.auto_residual_correction.active == Correction::JACOBI_DAVIDSON);
    REQUIRE(solver.auto_residual_correction.cheap_olsen_to_jd_switch_outer_iters.size() == 1);
}

TEST_CASE("standard auto residual correction separates directed Ritz progress from noise") {
    using Matrix     = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorReal = grit::form::base<double>::VectorReal;
    using Correction = grit::ResidualCorrectionType;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev                                 = 1;
    solver.config.ncv                                 = 4;
    solver.config.block_size                          = 1;
    solver.config.abstol                              = 1e-6;
    solver.config.residual_correction_type            = Correction::AUTO;
    solver.V                                          = Matrix::Identity(4, 1);
    solver.status.eigVal                              = VectorReal::Constant(1, 5.0);
    solver.status.rNormsAbs                           = VectorReal::Constant(1, 1e-3);
    solver.status.rNormsAbsHistory                    = {VectorReal::Constant(1, 1e-1), VectorReal::Constant(1, 1e-3), VectorReal::Constant(1, 1e-1),
                                                         VectorReal::Constant(1, 1e-3), VectorReal::Constant(1, 1e-1)};
    solver.auto_residual_correction.cheap_olsen_iters = 4;

    SECTION("clear monotonic motion keeps cheap Olsen") {
        solver.status.eigVals_history = {VectorReal::Constant(1, 1.0), VectorReal::Constant(1, 2.0), VectorReal::Constant(1, 3.0), VectorReal::Constant(1, 4.0),
                                         VectorReal::Constant(1, 5.0)};
        solver.update_auto_residual_correction_state();
        REQUIRE(solver.auto_residual_correction.active == Correction::CHEAP_OLSEN);
        REQUIRE(solver.auto_residual_correction.cheap_olsen_to_jd_switch_outer_iters.empty());
    }

    SECTION("a noisy plateau starts JD") {
        solver.status.eigVals_history = {VectorReal::Constant(1, 4.0), VectorReal::Constant(1, 6.0), VectorReal::Constant(1, 4.0), VectorReal::Constant(1, 6.0),
                                         VectorReal::Constant(1, 4.0)};
        solver.update_auto_residual_correction_state();
        REQUIRE(solver.auto_residual_correction.active == Correction::JACOBI_DAVIDSON);
    }

    SECTION("an isolated fluctuation starts JD") {
        solver.status.eigVals_history = {VectorReal::Constant(1, 5.0), VectorReal::Constant(1, 5.0), VectorReal::Constant(1, 5.0), VectorReal::Constant(1, 5.0),
                                         VectorReal::Constant(1, 6.0)};
        solver.update_auto_residual_correction_state();
        REQUIRE(solver.auto_residual_correction.active == Correction::JACOBI_DAVIDSON);
    }
}

TEST_CASE("standard auto residual correction evaluates every unconverged Ritz slot independently") {
    using Matrix     = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorReal = grit::form::base<double>::VectorReal;
    using Correction = grit::ResidualCorrectionType;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev                      = 2;
    solver.config.ncv                      = 4;
    solver.config.block_size               = 1;
    solver.config.abstol                   = 1e-6;
    solver.config.residual_correction_type = Correction::AUTO;
    solver.V                               = Matrix::Identity(4, 2);
    solver.status.eigVal                   = (VectorReal(2) << 1.0, 6.0).finished();
    solver.status.rNormsAbs                = VectorReal::Ones(2);
    solver.status.eigVals_history = {(VectorReal(2) << 1.0, 2.0).finished(), (VectorReal(2) << 1.0, 3.0).finished(), (VectorReal(2) << 1.0, 4.0).finished(),
                                     (VectorReal(2) << 1.0, 5.0).finished(), (VectorReal(2) << 1.0, 6.0).finished()};
    solver.auto_residual_correction.cheap_olsen_iters = 4;

    solver.update_auto_residual_correction_state();
    REQUIRE(solver.auto_residual_correction.active == Correction::CHEAP_OLSEN);

    solver.status.eigVals_history = {(VectorReal(2) << 1.0, 2.0).finished(), (VectorReal(2) << 1.0, 2.0).finished(), (VectorReal(2) << 1.0, 2.0).finished(),
                                     (VectorReal(2) << 1.0, 2.0).finished(), (VectorReal(2) << 1.0, 2.0).finished()};
    solver.update_auto_residual_correction_state();
    REQUIRE(solver.auto_residual_correction.active == Correction::JACOBI_DAVIDSON);
}

TEST_CASE("standard auto residual correction ignores converged Ritz slots in stabilization") {
    using Matrix     = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorReal = grit::form::base<double>::VectorReal;
    using Correction = grit::ResidualCorrectionType;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev                      = 2;
    solver.config.ncv                      = 4;
    solver.config.block_size               = 1;
    solver.config.abstol                   = 1e-6;
    solver.config.residual_correction_type = Correction::AUTO;
    solver.V                               = Matrix::Identity(4, 2);
    solver.status.eigVal                   = (VectorReal(2) << 5.0, 2.0).finished();
    solver.status.rNormsAbs                = (VectorReal(2) << 1e-8, 1.0).finished();
    solver.status.eigVals_history = {(VectorReal(2) << 1.0, 2.0).finished(), (VectorReal(2) << 2.0, 2.0).finished(), (VectorReal(2) << 3.0, 2.0).finished(),
                                     (VectorReal(2) << 4.0, 2.0).finished(), (VectorReal(2) << 5.0, 2.0).finished()};
    solver.auto_residual_correction.cheap_olsen_iters = 4;

    solver.update_auto_residual_correction_state();

    REQUIRE(solver.auto_residual_correction.active == Correction::JACOBI_DAVIDSON);
}

TEST_CASE("standard auto residual correction probe keeps Olsen when the rolling Ritz history is unstable") {
    using Matrix     = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorReal = grit::form::base<double>::VectorReal;
    using Correction = grit::ResidualCorrectionType;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev                                = 1;
    solver.config.ncv                                = 4;
    solver.config.block_size                         = 1;
    solver.config.abstol                             = 1e-6;
    solver.config.residual_correction_type           = Correction::AUTO;
    solver.V                                         = Matrix::Identity(4, 1);
    solver.status.rNormsAbs                          = VectorReal::Ones(1);
    solver.status.eigVal                             = VectorReal::Constant(1, 3.0);
    solver.status.eigVals_history                    = {VectorReal::Constant(1, 1.0), VectorReal::Constant(1, 2.0), VectorReal::Constant(1, 3.0)};
    solver.auto_residual_correction.active           = Correction::JACOBI_DAVIDSON;
    solver.auto_residual_correction.iteration_method = Correction::CHEAP_OLSEN;

    solver.update_auto_residual_correction_state();

    REQUIRE(solver.auto_residual_correction.active == Correction::JACOBI_DAVIDSON);
    REQUIRE(solver.auto_residual_correction.cheap_olsen_iters == 1);
    REQUIRE(solver.auto_residual_correction.probes_started == 1);
    REQUIRE(solver.auto_residual_correction.jd_outer_iters_since_probe == 0);

    solver.update_auto_residual_correction_state();

    REQUIRE(solver.auto_residual_correction.active == Correction::JACOBI_DAVIDSON);
    REQUIRE(solver.auto_residual_correction.cheap_olsen_iters == 2);

    solver.update_auto_residual_correction_state();

    REQUIRE(solver.auto_residual_correction.active == Correction::CHEAP_OLSEN);
    REQUIRE(solver.auto_residual_correction.cheap_olsen_iters == 3);
    REQUIRE(solver.auto_residual_correction.probes_started == 0);
    REQUIRE(solver.auto_residual_correction.jd_to_cheap_olsen_switch_outer_iters.size() == 1);

    solver.status.eigVals_history = {VectorReal::Constant(1, 3.0), VectorReal::Constant(1, 3.0), VectorReal::Constant(1, 3.0)};
    solver.update_auto_residual_correction_state();
    REQUIRE(solver.auto_residual_correction.active == Correction::JACOBI_DAVIDSON);
    REQUIRE(solver.auto_residual_correction.probes_started == 0);

    solver.status.rNormsAbsHistory                             = {VectorReal::Ones(1), VectorReal::Ones(1)};
    solver.auto_residual_correction.jd_outer_iters_since_probe = 2;
    solver.adjust_residual_correction_type();
    REQUIRE(solver.auto_residual_correction.iteration_method == Correction::CHEAP_OLSEN);
}

TEST_CASE("standard auto residual correction probe returns to JD on a noisy Ritz plateau") {
    using Matrix     = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorReal = grit::form::base<double>::VectorReal;
    using Correction = grit::ResidualCorrectionType;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev                      = 1;
    solver.config.ncv                      = 4;
    solver.config.block_size               = 1;
    solver.config.abstol                   = 1e-6;
    solver.config.residual_correction_type = Correction::AUTO;
    solver.V                               = Matrix::Identity(4, 1);
    solver.status.eigVal                   = VectorReal::Constant(1, 2.0);
    solver.status.rNormsAbs                = VectorReal::Ones(1);
    solver.status.eigVals_history = {VectorReal::Constant(1, 1.0), VectorReal::Constant(1, 3.0), VectorReal::Constant(1, 1.0), VectorReal::Constant(1, 3.0),
                                     VectorReal::Constant(1, 1.0)};
    solver.auto_residual_correction.active           = Correction::JACOBI_DAVIDSON;
    solver.auto_residual_correction.iteration_method = Correction::CHEAP_OLSEN;

    solver.update_auto_residual_correction_state();

    REQUIRE(solver.auto_residual_correction.active == Correction::JACOBI_DAVIDSON);
    REQUIRE(solver.auto_residual_correction.cheap_olsen_iters == 1);

    solver.update_auto_residual_correction_state();

    REQUIRE(solver.auto_residual_correction.active == Correction::JACOBI_DAVIDSON);
    REQUIRE(solver.auto_residual_correction.cheap_olsen_iters == 2);

    solver.update_auto_residual_correction_state();

    REQUIRE(solver.auto_residual_correction.active == Correction::JACOBI_DAVIDSON);
    REQUIRE(solver.auto_residual_correction.cheap_olsen_iters == 0);
    REQUIRE(solver.auto_residual_correction.probes_started == 1);
    REQUIRE(solver.auto_residual_correction.jd_to_cheap_olsen_switch_outer_iters.empty());

    solver.status.rNormsAbsHistory                             = {VectorReal::Ones(1), VectorReal::Ones(1)};
    solver.auto_residual_correction.jd_outer_iters_since_probe = 2;
    solver.adjust_residual_correction_type();
    REQUIRE(solver.auto_residual_correction.iteration_method == Correction::JACOBI_DAVIDSON);
}

TEST_CASE("standard gdplusk validates the Ritz stabilization tolerance") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev                          = 1;
    solver.config.ncv                          = 4;
    solver.config.block_size                   = 1;
    solver.config.ritz_stabilization_tolerance = 0;

    REQUIRE_THROWS_AS(solver.run(), std::runtime_error);
}

TEST_CASE("standard gdplusk validates the AUTO probe length") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev               = 1;
    solver.config.ncv               = 4;
    solver.config.block_size        = 1;
    solver.config.auto_probe_length = 0;

    REQUIRE_THROWS_AS(solver.run(), std::runtime_error);
}

TEST_CASE("standard gdplusk validates the AUTO maximum probes") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev             = 1;
    solver.config.ncv             = 4;
    solver.config.block_size      = 1;
    solver.config.auto_max_probes = -2;

    REQUIRE_THROWS_AS(solver.run(), std::runtime_error);
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

    using VectorReal         = grit::form::base<double>::VectorReal;
    solver.status.outer_iter = 2;
    solver.status.eigVal     = VectorReal::Constant(1, 1.0e6);
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

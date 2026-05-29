#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <grit/grit.h>

#include <Eigen/Eigenvalues>
#include <cmath>
#include <vector>

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

    auto solver = grit::standard::gdplusk<double>(A);
    solver.config.nev              = 2;
    solver.config.ncv              = A_matrix.rows();
    solver.config.block_size       = 1;
    solver.config.max_basis_blocks = A_matrix.rows();
    solver.config.ritz             = grit::OptRitz::SR;
    solver.config.max_iters        = 20;
    solver.set_initial_guess(V);
    solver.run();

    Eigen::SelfAdjointEigenSolver<Matrix> exact(A_matrix);
    auto view = grit::solver_view<double>(solver);
    REQUIRE(view.stopReason() == grit::StopReason::converged);
    require_close(view.eigVal(), exact.eigenvalues().head(2), 1e-10);
}

TEST_CASE("standard gdplusk converges with an exact zero eigenvalue") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(4, 4);
    A_matrix << 1.0, -1.0, 0.0, 0.0,
        -1.0, 2.0, -1.0, 0.0,
        0.0, -1.0, 2.0, -1.0,
        0.0, 0.0, -1.0, 1.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    Matrix V = Matrix::Identity(A_matrix.rows(), A_matrix.rows());

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev              = 1;
    solver.config.ncv              = A_matrix.rows();
    solver.config.block_size       = 1;
    solver.config.max_basis_blocks = A_matrix.rows();
    solver.config.ritz             = grit::OptRitz::SR;
    solver.config.max_iters        = 20;
    solver.config.tol              = 1e-12;
    solver.set_initial_guess(V);
    solver.run();

    auto view = grit::solver_view<double>(solver);
    REQUIRE(view.stopReason() == grit::StopReason::converged);
    REQUIRE(std::abs(view.eigVal()(0)) < 1e-12);
    REQUIRE(view.rNorms()(0) < 1e-12);
}

TEST_CASE("standard gdplusk user callback reports initial and final view") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 3.0, 0.5, 0.0, 0.0,
        0.0, 0.5, 2.0, 0.25, 0.0,
        0.0, 0.0, 0.25, 5.0, 0.5,
        0.0, 0.0, 0.0, 0.5, 6.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    Matrix V = Matrix::Identity(A_matrix.rows(), A_matrix.rows());

    std::vector<Eigen::Index> iterations;
    std::vector<grit::StopReason> stop_reasons;
    grit::standard::gdplusk<double> solver(A);
    solver.config.nev              = 1;
    solver.config.ncv              = A_matrix.rows();
    solver.config.block_size       = 1;
    solver.config.max_basis_blocks = A_matrix.rows();
    solver.config.ritz             = grit::OptRitz::SR;
    solver.config.max_iters        = 20;
    solver.config.user_callback    = [&](const auto &solver_ref) {
        auto view = grit::solver_view<double>(solver_ref);
        iterations.push_back(view.iter());
        stop_reasons.push_back(view.stopReason());
    };
    solver.set_initial_guess(V);
    solver.run();

    REQUIRE(iterations.size() >= 2);
    REQUIRE(iterations.front() == 0);
    REQUIRE(iterations.back() + 1 == grit::solver_view<double>(solver).iter());
    REQUIRE(stop_reasons.front() == grit::StopReason::none);
    REQUIRE(stop_reasons.back() == grit::StopReason::converged);
}

TEST_CASE("standard jacobi-davidson correction invokes preconditioner callbacks") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = grit::Matvec<double>::VectorType;

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

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev                      = 1;
    solver.config.ncv                      = 3;
    solver.config.block_size               = 1;
    solver.config.max_basis_blocks         = 3;
    solver.config.ritz                     = grit::OptRitz::SR;
    solver.config.max_iters                = 5;
    solver.config.inner_max_iters          = 20;
    solver.config.inner_tol                = 1e-8;
    solver.config.residual_correction_type = grit::ResidualCorrectionType::JACOBI_DAVIDSON;
    solver.set_initial_guess(V);
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

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev                      = 1;
    solver.config.ncv                      = 3;
    solver.config.block_size               = 1;
    solver.config.max_basis_blocks         = 3;
    solver.config.ritz                     = grit::OptRitz::SR;
    solver.config.max_iters                = 2;
    solver.config.inner_max_iters          = 20;
    solver.config.inner_tol                = 1e-8;
    solver.config.residual_correction_type = grit::ResidualCorrectionType::JACOBI_DAVIDSON;
    solver.set_initial_guess(V);

    REQUIRE_NOTHROW(solver.run());
}

TEST_CASE("standard auto residual correction starts with cheap Olsen") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev              = 1;
    solver.config.ncv              = 4;
    solver.config.block_size       = 1;
    solver.config.max_basis_blocks = 4;
    using Correction = grit::form::base<double>::ResidualCorrectionType;

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
    solver.config.nev              = 1;
    solver.config.ncv              = 4;
    solver.config.block_size       = 1;
    solver.config.max_basis_blocks = 4;
    using Base       = grit::form::base<double>;
    using Correction = Base::ResidualCorrectionType;
    using VectorReal = Base::VectorReal;

    solver.config.residual_correction_type       = Correction::AUTO;
    solver.auto_residual_correction.active       = Correction::CHEAP_OLSEN;
    solver.auto_residual_correction.step_method  = Correction::CHEAP_OLSEN;
    solver.auto_residual_correction.dwell        = solver.config.auto_min_dwell_iters;
    solver.status.iter                           = 2;
    solver.status.eigVal                         = VectorReal::Constant(1, -1.0);
    solver.status.rNorms                         = VectorReal::Constant(1, 1.0e-2);
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
    solver.config.nev              = 1;
    solver.config.ncv              = 4;
    solver.config.block_size       = 1;
    solver.config.max_basis_blocks = 4;
    using Base       = grit::form::base<double>;
    using VectorReal = Base::VectorReal;

    solver.config.auto_sat_eigval_threshold = 1.0e-3;
    solver.status.iter               = 2;
    solver.status.op_norm_estimate   = 1.0e6;
    solver.status.eigVal             = VectorReal::Constant(1, 1.01);
    solver.status.eigVals_history.clear();
    solver.status.eigVals_history.emplace_back(VectorReal::Constant(1, 1.00));
    solver.status.eigVals_history.emplace_back(VectorReal::Constant(1, 1.01));

    auto info = solver.get_auto_eigval_saturation_info();

    REQUIRE(info.enough_history);
    REQUIRE(info.scale == Approx(1.005));
    REQUIRE(info.ratio > solver.config.auto_sat_eigval_threshold);
    REQUIRE_FALSE(info.saturated);
}

TEST_CASE("standard auto residual correction honors cheap-Olsen dwell before Jacobi-Davidson") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev              = 1;
    solver.config.ncv              = 4;
    solver.config.block_size       = 1;
    solver.config.max_basis_blocks = 4;
    using Base       = grit::form::base<double>;
    using Correction = Base::ResidualCorrectionType;
    using VectorReal = Base::VectorReal;

    solver.config.residual_correction_type      = Correction::AUTO;
    solver.auto_residual_correction.active      = Correction::CHEAP_OLSEN;
    solver.auto_residual_correction.step_method = Correction::CHEAP_OLSEN;
    solver.auto_residual_correction.dwell       = 0;
    solver.status.iter                          = 2;
    solver.status.eigVal                        = VectorReal::Constant(1, -1.0);
    solver.status.rNorms                        = VectorReal::Constant(1, 1.0e-2);
    solver.status.eigVals_history.clear();
    solver.status.rNorms_history.clear();
    solver.status.eigVals_history.emplace_back(VectorReal::Constant(1, -1.0));
    solver.status.eigVals_history.emplace_back(VectorReal::Constant(1, -1.0 + 1.0e-8));
    solver.status.rNorms_history.emplace_back(VectorReal::Constant(1, 1.0e-2));
    solver.status.rNorms_history.emplace_back(VectorReal::Constant(1, 1.0e-2 + 1.0e-8));
    solver.status.num_matvecs       = 1;
    solver.status.num_matvecs_inner = 0;

    solver.update_auto_residual_correction_state();

    REQUIRE(solver.auto_residual_correction.active == Correction::CHEAP_OLSEN);
    REQUIRE(solver.auto_residual_correction.dwell == 1);
    REQUIRE(solver.auto_residual_correction.cheap_to_jd_switch_iters.empty());
}

TEST_CASE("standard auto residual correction starts Jacobi-Davidson below rrnorm threshold before cheap dwell") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev              = 1;
    solver.config.ncv              = 4;
    solver.config.block_size       = 1;
    solver.config.max_basis_blocks = 4;
    using Base       = grit::form::base<double>;
    using Correction = Base::ResidualCorrectionType;
    using VectorReal = Base::VectorReal;

    solver.config.residual_correction_type      = Correction::AUTO;
    solver.auto_residual_correction.active      = Correction::CHEAP_OLSEN;
    solver.auto_residual_correction.step_method = Correction::CHEAP_OLSEN;
    solver.auto_residual_correction.dwell       = 0;
    solver.config.auto_jd_start_rnorm_threshold = 1.0e-5;
    solver.status.iter                          = 2;
    solver.status.eigVal                        = VectorReal::Constant(1, -1.0);
    solver.status.rNorms                        = VectorReal::Constant(1, 1.0e-6);
    solver.status.num_matvecs                   = 1;
    solver.status.num_matvecs_inner             = 0;

    solver.update_auto_residual_correction_state();

    REQUIRE(solver.auto_residual_correction.active == Correction::JACOBI_DAVIDSON);
    REQUIRE(solver.auto_residual_correction.dwell == 0);
    REQUIRE(solver.auto_residual_correction.cheap_to_jd_switch_iters == std::vector<Eigen::Index>{2});
}

TEST_CASE("standard auto residual correction starts Jacobi-Davidson when histories saturate") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev              = 1;
    solver.config.ncv              = 4;
    solver.config.block_size       = 1;
    solver.config.max_basis_blocks = 4;
    using Base       = grit::form::base<double>;
    using Correction = Base::ResidualCorrectionType;
    using VectorReal = Base::VectorReal;

    solver.config.residual_correction_type      = Correction::AUTO;
    solver.auto_residual_correction.dwell       = solver.config.auto_min_dwell_iters;
    solver.config.auto_sat_eigval_threshold     = 1.0e-3;
    solver.config.auto_sat_rnorm_threshold      = 1.0e-2;
    solver.status.iter                          = 2;
    solver.status.op_norm_estimate              = 1.0;
    solver.status.eigVal                        = VectorReal::Constant(1, -1.0);
    solver.status.rNorms                        = VectorReal::Constant(1, 1.0e-2);
    solver.status.eigVals_history.clear();
    solver.status.rNorms_history.clear();
    solver.status.eigVals_history.emplace_back(VectorReal::Constant(1, -1.0));
    solver.status.eigVals_history.emplace_back(VectorReal::Constant(1, -1.0));
    solver.status.rNorms_history.emplace_back(VectorReal::Constant(1, 1.0e-2));
    solver.status.rNorms_history.emplace_back(VectorReal::Constant(1, 1.0e-2));
    solver.status.num_matvecs       = 1;
    solver.status.num_matvecs_inner              = 0;

    auto saturation = solver.get_auto_saturation_status();
    REQUIRE(saturation.eigval.saturated);
    REQUIRE(saturation.rnorm.saturated);

    solver.update_auto_residual_correction_state();

    REQUIRE(solver.auto_residual_correction.active == Correction::JACOBI_DAVIDSON);
    REQUIRE(solver.auto_residual_correction.cheap_to_jd_switch_iters == std::vector<Eigen::Index>{2});
    REQUIRE(solver.auto_residual_correction.jd_steps_since_probe == 0);
}

TEST_CASE("standard auto residual correction schedules cheap probe after interval") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev              = 1;
    solver.config.ncv              = 4;
    solver.config.block_size       = 1;
    solver.config.max_basis_blocks = 4;
    using Correction = grit::form::base<double>::ResidualCorrectionType;

    solver.config.residual_correction_type = Correction::AUTO;
    solver.auto_residual_correction.active               = Correction::JACOBI_DAVIDSON;
    solver.auto_residual_correction.jd_steps_since_probe = solver.config.auto_cheap_probe_interval;

    solver.adjust_residual_correction_type();

    REQUIRE(solver.residual_correction_type_internal == Correction::CHEAP_OLSEN);
    REQUIRE(solver.auto_residual_correction.step_method == Correction::CHEAP_OLSEN);
}

TEST_CASE("standard auto residual correction keeps cheap Olsen after useful probe") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev              = 1;
    solver.config.ncv              = 4;
    solver.config.block_size       = 1;
    solver.config.max_basis_blocks = 4;
    using Base       = grit::form::base<double>;
    using Correction = Base::ResidualCorrectionType;
    using VectorReal = Base::VectorReal;

    solver.config.residual_correction_type = Correction::AUTO;
    solver.status.oldVal                        = VectorReal::Constant(1, -1.0);
    solver.status.eigVal                        = VectorReal::Constant(1, -1.2);
    solver.status.rNorms                        = VectorReal::Constant(1, 0.1);
    solver.status.iter                          = 8;
    solver.status.num_matvecs                   = 1;
    solver.status.num_matvecs_inner              = 0;

    solver.auto_residual_correction.active      = Correction::JACOBI_DAVIDSON;
    solver.auto_residual_correction.step_method = Correction::CHEAP_OLSEN;
    solver.update_auto_residual_correction_state();

    REQUIRE(solver.auto_residual_correction.active == Correction::CHEAP_OLSEN);
    REQUIRE(solver.auto_residual_correction.jd_to_cheap_switch_iters == std::vector<Eigen::Index>{8});
}

TEST_CASE("standard auto residual correction returns to Jacobi-Davidson after weak cheap probe") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev              = 1;
    solver.config.ncv              = 4;
    solver.config.block_size       = 1;
    solver.config.max_basis_blocks = 4;
    using Base       = grit::form::base<double>;
    using Correction = Base::ResidualCorrectionType;
    using VectorReal = Base::VectorReal;

    solver.config.residual_correction_type       = Correction::AUTO;
    solver.auto_residual_correction.active       = Correction::JACOBI_DAVIDSON;
    solver.auto_residual_correction.step_method  = Correction::CHEAP_OLSEN;
    solver.auto_residual_correction.jd_steps_since_probe = solver.config.auto_cheap_probe_interval;
    solver.status.oldVal                         = VectorReal::Constant(1, -1.0);
    solver.status.eigVal                         = VectorReal::Constant(1, -1.005);
    solver.status.rNorms                         = VectorReal::Constant(1, 0.1);
    solver.status.iter                           = 9;
    solver.status.num_matvecs                    = 1;
    solver.status.num_matvecs_inner              = 0;

    solver.update_auto_residual_correction_state();

    REQUIRE(solver.auto_residual_correction.active == Correction::JACOBI_DAVIDSON);
    REQUIRE(solver.auto_residual_correction.jd_steps_since_probe == 0);
    REQUIRE(solver.auto_residual_correction.jd_to_cheap_switch_iters.empty());
}

TEST_CASE("standard auto residual correction cheap probe uses roundoff floor") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev              = 1;
    solver.config.ncv              = 4;
    solver.config.block_size       = 1;
    solver.config.max_basis_blocks = 4;
    using Base       = grit::form::base<double>;
    using Correction = Base::ResidualCorrectionType;
    using VectorReal = Base::VectorReal;

    solver.config.residual_correction_type                = Correction::AUTO;
    solver.auto_residual_correction.active                = Correction::JACOBI_DAVIDSON;
    solver.auto_residual_correction.step_method           = Correction::CHEAP_OLSEN;
    solver.auto_residual_correction.jd_steps_since_probe  = solver.config.auto_cheap_probe_interval;
    solver.status.oldVal                                  = VectorReal::Constant(1, 0.0);
    solver.status.eigVal                                  = VectorReal::Constant(1, -1.0e-18);
    solver.status.rNorms                                  = VectorReal::Constant(1, 0.0);
    solver.status.iter                                    = 10;
    solver.status.num_matvecs                             = 1;
    solver.status.num_matvecs_inner                       = 0;

    solver.update_auto_residual_correction_state();

    REQUIRE(solver.auto_residual_correction.active == Correction::JACOBI_DAVIDSON);
    REQUIRE(solver.auto_residual_correction.jd_steps_since_probe == 0);
    REQUIRE(solver.auto_residual_correction.jd_to_cheap_switch_iters.empty());
}

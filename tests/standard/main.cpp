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

    grit::standard::problem<double> problem(A);
    grit::gdplusk_config<double>    cfg;
    cfg.nev       = 2;
    cfg.ncv       = A_matrix.rows();
    cfg.block_size         = 1;
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
    cfg.block_size           = 1;
    cfg.ritz        = grit::OptRitz::SR;
    cfg.max_iters   = 5;
    cfg.inner_max_iters = 20;
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
    cfg.block_size           = 1;
    cfg.ritz        = grit::OptRitz::SR;
    cfg.max_iters   = 2;
    cfg.inner_max_iters = 20;
    cfg.inner_tol   = 1e-8;
    cfg.set_initial_guess(V);

    grit::standard::gdplusk<double> solver(problem, cfg);
    solver.residual_correction_type = grit::form::base<double>::ResidualCorrectionType::JACOBI_DAVIDSON;

    REQUIRE_NOTHROW(solver.run());
}

TEST_CASE("standard auto residual correction starts with cheap Olsen") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::problem<double> problem(A);
    grit::gdplusk_config<double>    cfg;
    cfg.nev = 1;
    cfg.ncv = 4;
    cfg.block_size   = 1;

    grit::standard::gdplusk<double> solver(problem, cfg);
    using Correction = grit::form::base<double>::ResidualCorrectionType;

    solver.residual_correction_type = Correction::AUTO;
    solver.adjust_residual_correction_type();

    REQUIRE(solver.residual_correction_type_internal == Correction::CHEAP_OLSEN);
    REQUIRE(solver.auto_residual_correction.active == Correction::CHEAP_OLSEN);
    REQUIRE(solver.auto_residual_correction.step_method == Correction::CHEAP_OLSEN);
}

TEST_CASE("standard auto residual correction does not start Jacobi-Davidson unless eigenvalues and residuals saturate") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::problem<double> problem(A);
    grit::gdplusk_config<double>    cfg;
    cfg.nev = 1;
    cfg.ncv = 4;
    cfg.block_size   = 1;

    grit::standard::gdplusk<double> solver(problem, cfg);
    using Base       = grit::form::base<double>;
    using Correction = Base::ResidualCorrectionType;
    using VectorReal = Base::VectorReal;

    solver.residual_correction_type              = Correction::AUTO;
    solver.auto_residual_correction.active       = Correction::CHEAP_OLSEN;
    solver.auto_residual_correction.step_method  = Correction::CHEAP_OLSEN;
    solver.auto_residual_correction.dwell        = solver.auto_min_dwell_iters;
    solver.status.iter                           = 2;
    solver.status.eigVal                         = VectorReal::Constant(1, -1.0);
    solver.status.rNorms                         = VectorReal::Constant(1, 1.0e-2);
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

    grit::standard::problem<double> problem(A);
    grit::gdplusk_config<double>    cfg;
    cfg.nev = 1;
    cfg.ncv = 4;
    cfg.block_size = 1;

    grit::standard::gdplusk<double> solver(problem, cfg);
    using Base       = grit::form::base<double>;
    using VectorReal = Base::VectorReal;

    solver.auto_sat_eigval_threshold = 1.0e-3;
    solver.status.iter               = 2;
    solver.status.op_norm_estimate   = 1.0e6;
    solver.status.eigVal             = VectorReal::Constant(1, 1.01);
    solver.status.eigVals_history.emplace_back(VectorReal::Constant(1, 1.00));
    solver.status.eigVals_history.emplace_back(VectorReal::Constant(1, 1.01));

    auto metric = solver.get_auto_eigval_saturation_metric();

    REQUIRE(metric.enough_history);
    REQUIRE(metric.scale == Approx(1.005));
    REQUIRE(metric.ratio > solver.auto_sat_eigval_threshold);
    REQUIRE_FALSE(metric.saturated);
}

TEST_CASE("standard auto residual correction honors cheap-Olsen dwell before Jacobi-Davidson") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::problem<double> problem(A);
    grit::gdplusk_config<double>    cfg;
    cfg.nev = 1;
    cfg.ncv = 4;
    cfg.block_size = 1;

    grit::standard::gdplusk<double> solver(problem, cfg);
    using Base       = grit::form::base<double>;
    using Correction = Base::ResidualCorrectionType;
    using VectorReal = Base::VectorReal;

    solver.residual_correction_type             = Correction::AUTO;
    solver.auto_residual_correction.active      = Correction::CHEAP_OLSEN;
    solver.auto_residual_correction.step_method = Correction::CHEAP_OLSEN;
    solver.auto_residual_correction.dwell       = 0;
    solver.status.iter                          = 2;
    solver.status.eigVal                        = VectorReal::Constant(1, -1.0);
    solver.status.rNorms                        = VectorReal::Constant(1, 1.0e-2);
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

TEST_CASE("standard auto residual correction starts Jacobi-Davidson below rnorm threshold before cheap dwell") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::problem<double> problem(A);
    grit::gdplusk_config<double>    cfg;
    cfg.nev = 1;
    cfg.ncv = 4;
    cfg.block_size = 1;

    grit::standard::gdplusk<double> solver(problem, cfg);
    using Base       = grit::form::base<double>;
    using Correction = Base::ResidualCorrectionType;
    using VectorReal = Base::VectorReal;

    solver.residual_correction_type             = Correction::AUTO;
    solver.auto_residual_correction.active      = Correction::CHEAP_OLSEN;
    solver.auto_residual_correction.step_method = Correction::CHEAP_OLSEN;
    solver.auto_residual_correction.dwell       = 0;
    solver.auto_jd_start_rnorm_threshold        = 1.0e-5;
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

    grit::standard::problem<double> problem(A);
    grit::gdplusk_config<double>    cfg;
    cfg.nev = 1;
    cfg.ncv = 4;
    cfg.block_size   = 1;

    grit::standard::gdplusk<double> solver(problem, cfg);
    using Base       = grit::form::base<double>;
    using Correction = Base::ResidualCorrectionType;
    using VectorReal = Base::VectorReal;

    solver.residual_correction_type = Correction::AUTO;
    solver.auto_residual_correction.dwell       = solver.auto_min_dwell_iters;
    solver.status.iter                          = 2;
    solver.status.eigVal                        = VectorReal::Constant(1, -1.0);
    solver.status.rNorms                        = VectorReal::Constant(1, 1.0e-2);
    solver.status.eigVals_history.emplace_back(VectorReal::Constant(1, -1.0));
    solver.status.eigVals_history.emplace_back(VectorReal::Constant(1, -1.0 + 1.0e-8));
    solver.status.rNorms_history.emplace_back(VectorReal::Constant(1, 1.0e-2));
    solver.status.rNorms_history.emplace_back(VectorReal::Constant(1, 1.0e-2 + 1.0e-8));
    solver.status.num_matvecs       = 1;
    solver.status.num_matvecs_inner              = 0;

    solver.update_auto_residual_correction_state();

    REQUIRE(solver.auto_residual_correction.active == Correction::JACOBI_DAVIDSON);
    REQUIRE(solver.auto_residual_correction.cheap_to_jd_switch_iters == std::vector<Eigen::Index>{2});
    REQUIRE(solver.auto_residual_correction.jd_steps_since_probe == 0);
}

TEST_CASE("standard auto residual correction schedules cheap probe after interval") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::problem<double> problem(A);
    grit::gdplusk_config<double>    cfg;
    cfg.nev = 1;
    cfg.ncv = 4;
    cfg.block_size   = 1;

    grit::standard::gdplusk<double> solver(problem, cfg);
    using Correction = grit::form::base<double>::ResidualCorrectionType;

    solver.residual_correction_type = Correction::AUTO;
    solver.auto_residual_correction.active               = Correction::JACOBI_DAVIDSON;
    solver.auto_residual_correction.jd_steps_since_probe = solver.auto_cheap_probe_interval;

    solver.adjust_residual_correction_type();

    REQUIRE(solver.residual_correction_type_internal == Correction::CHEAP_OLSEN);
    REQUIRE(solver.auto_residual_correction.step_method == Correction::CHEAP_OLSEN);
}

TEST_CASE("standard auto residual correction keeps cheap Olsen after useful probe") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(4, 4);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::problem<double> problem(A);
    grit::gdplusk_config<double>    cfg;
    cfg.nev = 1;
    cfg.ncv = 4;
    cfg.block_size   = 1;

    grit::standard::gdplusk<double> solver(problem, cfg);
    using Base       = grit::form::base<double>;
    using Correction = Base::ResidualCorrectionType;
    using VectorReal = Base::VectorReal;

    solver.residual_correction_type = Correction::AUTO;
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

    grit::standard::problem<double> problem(A);
    grit::gdplusk_config<double>    cfg;
    cfg.nev = 1;
    cfg.ncv = 4;
    cfg.block_size   = 1;

    grit::standard::gdplusk<double> solver(problem, cfg);
    using Base       = grit::form::base<double>;
    using Correction = Base::ResidualCorrectionType;
    using VectorReal = Base::VectorReal;

    solver.residual_correction_type              = Correction::AUTO;
    solver.auto_residual_correction.active       = Correction::JACOBI_DAVIDSON;
    solver.auto_residual_correction.step_method  = Correction::CHEAP_OLSEN;
    solver.auto_residual_correction.jd_steps_since_probe = solver.auto_cheap_probe_interval;
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

    grit::standard::problem<double> problem(A);
    grit::gdplusk_config<double>    cfg;
    cfg.nev = 1;
    cfg.ncv = 4;
    cfg.block_size = 1;

    grit::standard::gdplusk<double> solver(problem, cfg);
    using Base       = grit::form::base<double>;
    using Correction = Base::ResidualCorrectionType;
    using VectorReal = Base::VectorReal;

    solver.residual_correction_type                       = Correction::AUTO;
    solver.auto_residual_correction.active                = Correction::JACOBI_DAVIDSON;
    solver.auto_residual_correction.step_method           = Correction::CHEAP_OLSEN;
    solver.auto_residual_correction.jd_steps_since_probe  = solver.auto_cheap_probe_interval;
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

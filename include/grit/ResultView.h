#pragma once

#include "enums.h"
#include <complex>
#include <Eigen/Core>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace grit::form {
    template<typename Scalar, grit::Form form>
    class base;
}

namespace grit {
    template<typename Scalar>
    class Result;

    /*! Non-owning read-only view of eigenpairs, residuals, counters, and timings from a solver. */
    template<typename Scalar_>
    class ResultView {
        public:
        using Scalar     = Scalar_;                                               /*!< Scalar type of the solver. */
        using RealScalar = decltype(std::real(std::declval<Scalar>()));           /*!< Real scalar type used for Ritz values and norms. */
        using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>; /*!< Dense block of eigenvectors. */
        using VectorReal = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;          /*!< Real-valued vector. */

        ResultView() = default;
        /*!
         * View the current state of a standard or generalized solver form.
         * @param source_ Solver form to view.
         */
        template<grit::Form form>
        explicit ResultView(const form::base<Scalar, form> &source_);

        /*! Selected Ritz values. */
        [[nodiscard]] const VectorReal &eigVal() const;
        /*! Selected Ritz vectors in columns. */
        [[nodiscard]] const MatrixType &eigVecs() const;
        /*! Residual norms for the selected Ritz pairs. */
        [[nodiscard]] const VectorReal &rNormsAbs() const;
        /*! Reason why the solver stopped. */
        [[nodiscard]] StopReason stopReason() const;
        /*! Current outer iteration count. */
        [[nodiscard]] Eigen::Index outer_iter() const;
        /*! Matrix-vector products in the last outer iteration. */
        [[nodiscard]] Eigen::Index num_matvecs() const;
        /*! Inner correction iterations in the last outer iteration. */
        [[nodiscard]] Eigen::Index num_inner_iters() const;
        /*! Matrix-vector products in the last inner correction solve. */
        [[nodiscard]] Eigen::Index num_matvecs_inner() const;
        /*! A matrix-vector products in the last outer iteration. */
        [[nodiscard]] Eigen::Index num_matvecs_a() const;
        /*! B matrix-vector products in the last outer iteration. */
        [[nodiscard]] Eigen::Index num_matvecs_b() const;
        /*! A matrix-vector products in the last inner correction solve. */
        [[nodiscard]] Eigen::Index num_matvecs_a_inner() const;
        /*! B matrix-vector products in the last inner correction solve. */
        [[nodiscard]] Eigen::Index num_matvecs_b_inner() const;
        /*! Total A matrix-vector products. */
        [[nodiscard]] Eigen::Index num_matvecs_a_total() const;
        /*! Total B matrix-vector products. */
        [[nodiscard]] Eigen::Index num_matvecs_b_total() const;
        /*! Projected correction-operator applications in the last inner solve. */
        [[nodiscard]] Eigen::Index num_operator_inner() const;
        /*! Total matrix-vector products. */
        [[nodiscard]] Eigen::Index num_matvecs_total() const;
        /*! Preconditioner applications in the last outer iteration. */
        [[nodiscard]] Eigen::Index num_precond() const;
        /*! Projected-preconditioner applications in the last inner correction solve, including identity applications. */
        [[nodiscard]] Eigen::Index num_precond_inner() const;
        /*! Total preconditioner applications. */
        [[nodiscard]] Eigen::Index num_precond_total() const;
        /*! Preconditioner update callbacks in the last outer iteration. */
        [[nodiscard]] Eigen::Index num_preconditioner_updates() const;
        /*! Preconditioner update callbacks for the last inner correction. */
        [[nodiscard]] Eigen::Index num_preconditioner_updates_inner() const;
        /*! Total preconditioner update callbacks. */
        [[nodiscard]] Eigen::Index num_preconditioner_updates_total() const;
        /*! User preconditioner apply callbacks in the last inner solve. */
        [[nodiscard]] Eigen::Index num_preconditioner_apply_inner() const;
        /*! Total user preconditioner apply callbacks in inner solves. */
        [[nodiscard]] Eigen::Index num_preconditioner_apply_inner_total() const;
        /*! Total user preconditioner apply callbacks. */
        [[nodiscard]] Eigen::Index num_preconditioner_apply_total() const;
        /*! Total wall time measured by the solver. */
        [[nodiscard]] RealScalar time() const;
        /*! Complete wall time spent in inner correction solves. */
        [[nodiscard]] RealScalar time_solve_inner() const;
        /*! Inclusive projected correction-operator time. */
        [[nodiscard]] RealScalar time_operator_inner() const;
        /*! Outer A and B callback time. */
        [[nodiscard]] RealScalar time_matvecs() const;
        /*! Inner callback time, computed as A callback time plus B callback time. */
        [[nodiscard]] RealScalar time_matvecs_inner() const;
        /*! Outer A callback time. */
        [[nodiscard]] RealScalar time_matvecs_a() const;
        /*! Outer B callback time. */
        [[nodiscard]] RealScalar time_matvecs_b() const;
        /*! Inner A callback time. */
        [[nodiscard]] RealScalar time_matvecs_a_inner() const;
        /*! Inner B callback time. */
        [[nodiscard]] RealScalar time_matvecs_b_inner() const;
        /*! Inclusive projected-preconditioner time. */
        [[nodiscard]] RealScalar time_preconditioner_inner() const;
        /*! Outer preconditioner update callback time. */
        [[nodiscard]] RealScalar time_preconditioner_update() const;
        /*! Inner preconditioner update callback time. */
        [[nodiscard]] RealScalar time_preconditioner_update_inner() const;
        /*! Total preconditioner update callback time. */
        [[nodiscard]] RealScalar time_preconditioner_update_total() const;
        /*! Inner user preconditioner apply callback time. */
        [[nodiscard]] RealScalar time_preconditioner_apply_inner() const;
        /*! Total user preconditioner apply callback time. */
        [[nodiscard]] RealScalar time_preconditioner_apply_total() const;
        /*! Inner left-projector time. */
        [[nodiscard]] RealScalar time_project_left_inner() const;
        /*! Inner right-projector time. */
        [[nodiscard]] RealScalar time_project_right_inner() const;
        /*! Residual-correction construction time. */
        [[nodiscard]] RealScalar time_residual_correction() const;
        /*! Search-space build time. */
        [[nodiscard]] RealScalar time_build() const;
        /*! Convergence and status-update time. */
        [[nodiscard]] RealScalar time_status_update() const;
        /*! Time spent orthogonalizing new vectors. */
        [[nodiscard]] RealScalar time_orthogonalize() const;
        /*! Time spent orthonormalizing new vectors. */
        [[nodiscard]] RealScalar time_orthonormalize() const;
        /*! Time spent projecting during orthogonalization. */
        [[nodiscard]] RealScalar time_orth_project() const;
        /*! Time spent factorizing during orthonormalization. */
        [[nodiscard]] RealScalar time_orth_factor() const;
        /*! Time spent updating orthogonalized vectors. */
        [[nodiscard]] RealScalar time_orth_update() const;
        /*! Time spent refreshing operator products after orthogonalization. */
        [[nodiscard]] RealScalar time_orth_refresh() const;
        /*! Time spent masking dependent vectors. */
        [[nodiscard]] RealScalar time_orth_mask() const;
        /*! Time spent diagonalizing projected problems. */
        [[nodiscard]] RealScalar time_diagonalize() const;
        /*! Time spent extracting Ritz vectors and residuals. */
        [[nodiscard]] RealScalar time_extract_ritz() const;
        /*! Time spent restarting the search space. */
        [[nodiscard]] RealScalar time_restart() const;
        /*! Last inner correction residual. */
        [[nodiscard]] RealScalar inner_error_last() const;
        /*! Last inner correction tolerance. */
        [[nodiscard]] RealScalar inner_tol_last() const;
        /*! Number of consecutive eigenvalue saturation checks. */
        [[nodiscard]] Eigen::Index saturation_count_eigval() const;
        /*! Number of consecutive residual saturation checks. */
        [[nodiscard]] Eigen::Index saturation_count_rnorm() const;
        /*! Saturation count required before a saturation stop. */
        [[nodiscard]] Eigen::Index saturation_count_max() const;
        /*! Current operator norm estimate used for rescaled residuals. */
        [[nodiscard]] RealScalar op_norm_estimate() const;
        /*! Current projected-problem condition estimate. */
        [[nodiscard]] RealScalar condition() const;
        /*! Current eigenvalue sensitivity estimate. */
        [[nodiscard]] RealScalar sensitivity() const;
        /*! Current Ritz gap estimate. */
        [[nodiscard]] RealScalar gap() const;
        /*! Whether all selected residual norms are below tolerance. */
        [[nodiscard]] bool residual_converged() const;
        /*! Whether all selected residual norms are below the Ritz gap criterion. */
        [[nodiscard]] bool residual_below_gap() const;
        /*! AUTO residual correction method currently in use. */
        [[nodiscard]] std::string_view residual_correction_active_name() const;
        /*! Residual correction method used in the last outer iteration. */
        [[nodiscard]] std::string_view residual_correction_iteration_name() const;
        /*! Consecutive AUTO outer iterations spent in cheap Olsen mode. */
        [[nodiscard]] Eigen::Index auto_cheap_olsen_iters() const;
        /*! Number of outer iterations using JD corrections since the last cheap Olsen correction. */
        [[nodiscard]] Eigen::Index auto_jd_outer_iters_since_probe() const;
        /*! Iterations where AUTO switched from cheap Olsen to Jacobi-Davidson. */
        [[nodiscard]] const std::vector<Eigen::Index> &cheap_olsen_to_jd_switch_outer_iters() const;
        /*! Iterations where AUTO switched from Jacobi-Davidson to cheap Olsen. */
        [[nodiscard]] const std::vector<Eigen::Index> &jd_to_cheap_olsen_switch_outer_iters() const;
        /*! Copy this view into an owning result snapshot. */
        [[nodiscard]] Result<Scalar> to_result() const;

        private:
        std::variant<const form::base<Scalar, grit::Form::STANDARD> *, const form::base<Scalar, grit::Form::GENERALIZED> *> source = nullptr;
    };
}

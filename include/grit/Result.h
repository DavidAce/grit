#pragma once

#include "ResultView.h"
#include <string>
#include <vector>

namespace grit {
    /*! Owning snapshot of solver eigenpairs, residuals, counters, and timings. */
    template<typename Scalar_>
    class Result {
        public:
        using Scalar     = Scalar_;                                               /*!< Scalar type of the solver. */
        using RealScalar = decltype(std::real(std::declval<Scalar>()));           /*!< Real scalar type used for Ritz values and norms. */
        using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>; /*!< Dense block of eigenvectors. */
        using VectorReal = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;          /*!< Real-valued vector. */

        Result() = default;
        /*!
         * Copy an owning snapshot from a non-owning result view.
         * @param view Solver result view to copy from.
         */
        explicit Result(const ResultView<Scalar> &view)
            : eigVal_(view.eigVal()), eigVecs_(view.eigVecs()), rNorms_(view.rNormsAbs()), stopReason_(view.stopReason()), outer_iter_(view.outer_iter()),
              num_matvecs_(view.num_matvecs()), num_inner_iters_(view.num_inner_iters()), num_matvecs_inner_(view.num_matvecs_inner()),
              num_jdops_inner_(view.num_jdops_inner()), num_matvecs_total_(view.num_matvecs_total()), num_precond_(view.num_precond()),
              num_precond_inner_(view.num_precond_inner()), num_precond_total_(view.num_precond_total()), time_(view.time()),
              time_orthogonalize_(view.time_orthogonalize()), time_orthonormalize_(view.time_orthonormalize()),
              time_orth_project_(view.time_orth_project()), time_orth_factor_(view.time_orth_factor()),
              time_orth_update_(view.time_orth_update()), time_orth_refresh_(view.time_orth_refresh()), time_orth_mask_(view.time_orth_mask()),
              time_diagonalize_(view.time_diagonalize()), time_extract_ritz_(view.time_extract_ritz()), time_restart_(view.time_restart()),
              inner_error_last_(view.inner_error_last()), inner_tol_last_(view.inner_tol_last()), saturation_count_eigval_(view.saturation_count_eigval()),
              saturation_count_rnorm_(view.saturation_count_rnorm()), saturation_count_max_(view.saturation_count_max()),
              op_norm_estimate_(view.op_norm_estimate()), condition_(view.condition()), sensitivity_(view.sensitivity()), gap_(view.gap()),
              residual_converged_(view.residual_converged()), residual_below_gap_(view.residual_below_gap()),
              residual_correction_active_name_(view.residual_correction_active_name()),
              residual_correction_iteration_name_(view.residual_correction_iteration_name()), auto_dwell_(view.auto_dwell()),
              auto_jd_outer_iters_since_probe_(view.auto_jd_outer_iters_since_probe()),
              cheap_to_jd_switch_outer_iters_(view.cheap_to_jd_switch_outer_iters()),
              jd_to_cheap_switch_outer_iters_(view.jd_to_cheap_switch_outer_iters()) {}

        /*! Selected Ritz values. */
        [[nodiscard]] const VectorReal &eigVal() const { return eigVal_; }
        /*! Selected Ritz values. */
        [[nodiscard]] VectorReal &eigVal() { return eigVal_; }
        /*! Selected Ritz vectors in columns. */
        [[nodiscard]] const MatrixType &eigVecs() const { return eigVecs_; }
        /*! Selected Ritz vectors in columns. */
        [[nodiscard]] MatrixType &eigVecs() { return eigVecs_; }
        /*! Residual norms for the selected Ritz pairs. */
        [[nodiscard]] const VectorReal &rNormsAbs() const { return rNorms_; }
        /*! Residual norms for the selected Ritz pairs. */
        [[nodiscard]] VectorReal &rNormsAbs() { return rNorms_; }
        /*! Reason why the solver stopped. */
        [[nodiscard]] StopReason stopReason() const { return stopReason_; }
        /*! Current outer iteration count. */
        [[nodiscard]] Eigen::Index outer_iter() const { return outer_iter_; }
        /*! Matrix-vector products in the last outer iteration. */
        [[nodiscard]] Eigen::Index num_matvecs() const { return num_matvecs_; }
        /*! Inner correction iterations in the last outer iteration. */
        [[nodiscard]] Eigen::Index num_inner_iters() const { return num_inner_iters_; }
        /*! Matrix-vector products in the last inner correction solve. */
        [[nodiscard]] Eigen::Index num_matvecs_inner() const { return num_matvecs_inner_; }
        /*! Jacobi-Davidson operator applications in the last inner solve. */
        [[nodiscard]] Eigen::Index num_jdops_inner() const { return num_jdops_inner_; }
        /*! Total matrix-vector products. */
        [[nodiscard]] Eigen::Index num_matvecs_total() const { return num_matvecs_total_; }
        /*! Preconditioner applications in the last outer iteration. */
        [[nodiscard]] Eigen::Index num_precond() const { return num_precond_; }
        /*! Preconditioner applications in the last inner correction solve. */
        [[nodiscard]] Eigen::Index num_precond_inner() const { return num_precond_inner_; }
        /*! Total preconditioner applications. */
        [[nodiscard]] Eigen::Index num_precond_total() const { return num_precond_total_; }
        /*! Total wall time measured by the solver. */
        [[nodiscard]] RealScalar time() const { return time_; }
        /*! Time spent orthogonalizing new vectors. */
        [[nodiscard]] RealScalar time_orthogonalize() const { return time_orthogonalize_; }
        /*! Time spent orthonormalizing new vectors. */
        [[nodiscard]] RealScalar time_orthonormalize() const { return time_orthonormalize_; }
        /*! Time spent projecting during orthogonalization. */
        [[nodiscard]] RealScalar time_orth_project() const { return time_orth_project_; }
        /*! Time spent factorizing during orthonormalization. */
        [[nodiscard]] RealScalar time_orth_factor() const { return time_orth_factor_; }
        /*! Time spent updating orthogonalized vectors. */
        [[nodiscard]] RealScalar time_orth_update() const { return time_orth_update_; }
        /*! Time spent refreshing operator products after orthogonalization. */
        [[nodiscard]] RealScalar time_orth_refresh() const { return time_orth_refresh_; }
        /*! Time spent masking dependent vectors. */
        [[nodiscard]] RealScalar time_orth_mask() const { return time_orth_mask_; }
        /*! Time spent diagonalizing projected problems. */
        [[nodiscard]] RealScalar time_diagonalize() const { return time_diagonalize_; }
        /*! Time spent extracting Ritz vectors and residuals. */
        [[nodiscard]] RealScalar time_extract_ritz() const { return time_extract_ritz_; }
        /*! Time spent restarting the search space. */
        [[nodiscard]] RealScalar time_restart() const { return time_restart_; }
        /*! Last inner correction residual. */
        [[nodiscard]] RealScalar inner_error_last() const { return inner_error_last_; }
        /*! Last inner correction tolerance. */
        [[nodiscard]] RealScalar inner_tol_last() const { return inner_tol_last_; }
        /*! Number of consecutive eigenvalue saturation checks. */
        [[nodiscard]] Eigen::Index saturation_count_eigval() const { return saturation_count_eigval_; }
        /*! Number of consecutive residual saturation checks. */
        [[nodiscard]] Eigen::Index saturation_count_rnorm() const { return saturation_count_rnorm_; }
        /*! Saturation count required before a saturation stop. */
        [[nodiscard]] Eigen::Index saturation_count_max() const { return saturation_count_max_; }
        /*! Current operator norm estimate used for rescaled residuals. */
        [[nodiscard]] RealScalar op_norm_estimate() const { return op_norm_estimate_; }
        /*! Current projected-problem condition estimate. */
        [[nodiscard]] RealScalar condition() const { return condition_; }
        /*! Current eigenvalue sensitivity estimate. */
        [[nodiscard]] RealScalar sensitivity() const { return sensitivity_; }
        /*! Current Ritz gap estimate. */
        [[nodiscard]] RealScalar gap() const { return gap_; }
        /*! Whether all selected residual norms are below tolerance. */
        [[nodiscard]] bool residual_converged() const { return residual_converged_; }
        /*! Whether all selected residual norms are below the Ritz gap criterion. */
        [[nodiscard]] bool residual_below_gap() const { return residual_below_gap_; }
        /*! AUTO residual correction method currently in use. */
        [[nodiscard]] std::string_view residual_correction_active_name() const { return residual_correction_active_name_; }
        /*! Residual correction method used in the last outer iteration. */
        [[nodiscard]] std::string_view residual_correction_iteration_name() const { return residual_correction_iteration_name_; }
        /*! AUTO dwell outer iterations spent in cheap Olsen mode. */
        [[nodiscard]] Eigen::Index auto_dwell() const { return auto_dwell_; }
        /*! AUTO Jacobi-Davidson outer iterations since the last cheap Olsen probe. */
        [[nodiscard]] Eigen::Index auto_jd_outer_iters_since_probe() const { return auto_jd_outer_iters_since_probe_; }
        /*! Iterations where AUTO switched from cheap Olsen to Jacobi-Davidson. */
        [[nodiscard]] const std::vector<Eigen::Index> &cheap_to_jd_switch_outer_iters() const { return cheap_to_jd_switch_outer_iters_; }
        /*! Iterations where AUTO switched from Jacobi-Davidson to cheap Olsen. */
        [[nodiscard]] const std::vector<Eigen::Index> &jd_to_cheap_switch_outer_iters() const { return jd_to_cheap_switch_outer_iters_; }

        private:
        VectorReal                eigVal_;
        MatrixType                eigVecs_;
        VectorReal                rNorms_;
        StopReason                stopReason_ = StopReason::none;
        Eigen::Index              outer_iter_ = 0;
        Eigen::Index              num_matvecs_ = 0;
        Eigen::Index              num_inner_iters_ = 0;
        Eigen::Index              num_matvecs_inner_ = 0;
        Eigen::Index              num_jdops_inner_ = 0;
        Eigen::Index              num_matvecs_total_ = 0;
        Eigen::Index              num_precond_ = 0;
        Eigen::Index              num_precond_inner_ = 0;
        Eigen::Index              num_precond_total_ = 0;
        RealScalar                time_ = RealScalar{0};
        RealScalar                time_orthogonalize_ = RealScalar{0};
        RealScalar                time_orthonormalize_ = RealScalar{0};
        RealScalar                time_orth_project_ = RealScalar{0};
        RealScalar                time_orth_factor_ = RealScalar{0};
        RealScalar                time_orth_update_ = RealScalar{0};
        RealScalar                time_orth_refresh_ = RealScalar{0};
        RealScalar                time_orth_mask_ = RealScalar{0};
        RealScalar                time_diagonalize_ = RealScalar{0};
        RealScalar                time_extract_ritz_ = RealScalar{0};
        RealScalar                time_restart_ = RealScalar{0};
        RealScalar                inner_error_last_ = RealScalar{0};
        RealScalar                inner_tol_last_ = RealScalar{0};
        Eigen::Index              saturation_count_eigval_ = 0;
        Eigen::Index              saturation_count_rnorm_ = 0;
        Eigen::Index              saturation_count_max_ = 0;
        RealScalar                op_norm_estimate_ = RealScalar{0};
        RealScalar                condition_ = RealScalar{0};
        RealScalar                sensitivity_ = RealScalar{0};
        RealScalar                gap_ = RealScalar{0};
        bool                      residual_converged_ = false;
        bool                      residual_below_gap_ = false;
        std::string               residual_correction_active_name_;
        std::string               residual_correction_iteration_name_;
        Eigen::Index              auto_dwell_ = 0;
        Eigen::Index              auto_jd_outer_iters_since_probe_ = 0;
        std::vector<Eigen::Index> cheap_to_jd_switch_outer_iters_;
        std::vector<Eigen::Index> jd_to_cheap_switch_outer_iters_;
    };

    template<typename Scalar_>
    Result<Scalar_> ResultView<Scalar_>::to_result() const {
        return Result<Scalar_>(*this);
    }
}

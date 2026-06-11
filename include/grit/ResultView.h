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
    /*! Read-only view of the eigenpairs, residuals, counters, and timings from a solver. */
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
        [[nodiscard]] const VectorReal &rNorms() const;
        /*! Reason why the solver stopped. */
        [[nodiscard]] StopReason stopReason() const;
        /*! Current outer iteration count. */
        [[nodiscard]] Eigen::Index iter() const;
        /*! Matrix-vector products in the last outer step. */
        [[nodiscard]] Eigen::Index num_matvecs() const;
        /*! Inner correction iterations in the last outer step. */
        [[nodiscard]] Eigen::Index num_iters_inner() const;
        /*! Matrix-vector products in the last inner correction solve. */
        [[nodiscard]] Eigen::Index num_matvecs_inner() const;
        /*! Jacobi-Davidson operator applications in the last inner solve. */
        [[nodiscard]] Eigen::Index num_jdops_inner() const;
        /*! Total matrix-vector products. */
        [[nodiscard]] Eigen::Index num_matvecs_total() const;
        /*! Preconditioner applications in the last outer step. */
        [[nodiscard]] Eigen::Index num_precond() const;
        /*! Preconditioner applications in the last inner correction solve. */
        [[nodiscard]] Eigen::Index num_precond_inner() const;
        /*! Total preconditioner applications. */
        [[nodiscard]] Eigen::Index num_precond_total() const;
        /*! Total wall time measured by the solver. */
        [[nodiscard]] RealScalar seconds() const;
        /*! Time spent orthogonalizing new vectors. */
        [[nodiscard]] RealScalar seconds_orthogonalize() const;
        /*! Time spent orthonormalizing new vectors. */
        [[nodiscard]] RealScalar seconds_orthonormalize() const;
        /*! Time spent projecting during orthogonalization. */
        [[nodiscard]] RealScalar seconds_orth_project() const;
        /*! Time spent factorizing during orthonormalization. */
        [[nodiscard]] RealScalar seconds_orth_factor() const;
        /*! Time spent updating orthogonalized vectors. */
        [[nodiscard]] RealScalar seconds_orth_update() const;
        /*! Time spent refreshing operator products after orthogonalization. */
        [[nodiscard]] RealScalar seconds_orth_refresh() const;
        /*! Time spent masking dependent vectors. */
        [[nodiscard]] RealScalar seconds_orth_mask() const;
        /*! Time spent diagonalizing projected problems. */
        [[nodiscard]] RealScalar seconds_diagonalize() const;
        /*! Time spent extracting Ritz vectors and residuals. */
        [[nodiscard]] RealScalar seconds_extract_ritz() const;
        /*! Time spent restarting the search space. */
        [[nodiscard]] RealScalar seconds_restart() const;
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
        /*! Current operator norm estimate used for relative residuals. */
        [[nodiscard]] RealScalar op_norm_estimate() const;
        /*! Current projected-problem condition estimate. */
        [[nodiscard]] RealScalar condition() const;
        /*! Current eigenvalue sensitivity estimate. */
        [[nodiscard]] RealScalar sensitivity() const;
        /*! Current Ritz gap estimate. */
        [[nodiscard]] RealScalar gap() const;
        /*! Whether all selected residual norms are below tolerance. */
        [[nodiscard]] bool rnorm_below_tol() const;
        /*! Whether all selected residual norms are below the Ritz gap criterion. */
        [[nodiscard]] bool rnorm_below_gap() const;
        /*! AUTO residual correction method currently in use. */
        [[nodiscard]] std::string_view residual_correction_active_name() const;
        /*! Residual correction method used in the last step. */
        [[nodiscard]] std::string_view residual_correction_step_name() const;
        /*! AUTO dwell iterations spent in cheap Olsen mode. */
        [[nodiscard]] Eigen::Index auto_dwell() const;
        /*! AUTO Jacobi-Davidson steps since the last cheap Olsen probe. */
        [[nodiscard]] Eigen::Index auto_jd_steps_since_probe() const;
        /*! Iterations where AUTO switched from cheap Olsen to Jacobi-Davidson. */
        [[nodiscard]] const std::vector<Eigen::Index> &cheap_to_jd_switch_iters() const;
        /*! Iterations where AUTO switched from Jacobi-Davidson to cheap Olsen. */
        [[nodiscard]] const std::vector<Eigen::Index> &jd_to_cheap_switch_iters() const;

        private:
        std::variant<const form::base<Scalar, grit::Form::STANDARD> *, const form::base<Scalar, grit::Form::GENERALIZED> *> source = nullptr;
    };
}

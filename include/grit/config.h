#pragma once

#include "enums.h"
#include <complex>
#include <Eigen/Core>
#include <limits>
#include <optional>
#include <spdlog/common.h>
#include <utility>

namespace grit {
    template<typename Scalar_>
    class gdplusk_config {
        public:
        using Scalar     = Scalar_;
        using RealScalar = decltype(std::real(std::declval<Scalar>()));
        using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

        Eigen::Index              nev                         = 1;                                      /*!< Number of requested eigenpairs. */
        Eigen::Index              ncv                         = 8;                                      /*!< Maximum search-space columns; negative derives it from max_basis_blocks * block_size. */
        Eigen::Index              block_size                  = 2;                                      /*!< Number of vectors in each solver block. */
        OptRitz                   ritz                        = OptRitz::SR;                            /*!< Ritz target selection. */
        Eigen::Index              max_iters                   = 100;                                    /*!< Maximum outer solver iterations; negative means unlimited. */
        Eigen::Index              max_matvecs                 = -1;                                     /*!< Maximum total matrix-vector products; negative means unlimited. */
        RealScalar                tol                         = std::numeric_limits<RealScalar>::epsilon() * 10000; /*!< Absolute residual-norm convergence tolerance. */
        RealScalar                tol_rnorm_relative          = 0;                                      /*!< Relative residual-norm convergence tolerance; zero disables it. */
        RealScalar                sat_eigval_threshold        = RealScalar{0};                          /*!< Eigenvalue saturation threshold for stopping; zero disables this stop. */
        RealScalar                sat_rnorm_threshold         = RealScalar{0};                          /*!< Residual-norm saturation threshold for stopping; zero disables this stop. */
        RealScalar                inner_tol                   = RealScalar{0.1f};                       /*!< Target residual reduction for the inner correction solver. */
        Eigen::Index              inner_max_iters             = 1000;                                   /*!< Maximum iterations for the inner correction solver. */
        Eigen::Index              auto_min_dwell_iters        = 10;                                     /*!< Minimum consecutive cheap-Olsen AUTO steps before JD activation may occur. */
        RealScalar                auto_sat_eigval_threshold   = RealScalar{1e-3f};                      /*!< Eigenvalue saturation threshold for AUTO JD activation. */
        RealScalar                auto_sat_rnorm_threshold    = RealScalar{1e-2f};                      /*!< Residual-norm saturation threshold for AUTO JD activation. */
        RealScalar                auto_jd_start_rnorm_threshold = RealScalar{1e-5f};                    /*!< Residual norm below which AUTO may activate JD; zero disables it. */
        Eigen::Index              auto_cheap_probe_interval   = 5;                                      /*!< JD steps before AUTO forces a cheap-Olsen probe. */
        RealScalar                auto_cheap_probe_factor     = RealScalar{1.0f};                       /*!< Cheap probe must improve the Ritz value by this factor times max(rnorm^2, roundoff scale). */
        spdlog::level::level_enum log_level                    = spdlog::level::warn;                    /*!< Logger verbosity. */

        Eigen::Index max_extra_ritz_history    = 1; /*!< Number of previous Ritz-vector blocks retained during restart. */
        Eigen::Index max_ritz_residual_history = 1; /*!< Number of Ritz-residual blocks retained during restart. */
        Eigen::Index max_basis_blocks          = 8; /*!< Maximum number of block_size column groups in the search basis. */
        Eigen::Index max_retain_blocks         = 1; /*!< Maximum number of existing basis blocks retained during restart. */

        gdplusk_config() = default;

        gdplusk_config &set_initial_guess(const MatrixType &V);
        gdplusk_config &clear_initial_guess();

        [[nodiscard]] bool              has_initial_guess() const;
        [[nodiscard]] const MatrixType &initial_guess() const;

        private:
        const MatrixType *initial_guess_ptr = nullptr;
        MatrixType        empty_initial_guess;
    };
}

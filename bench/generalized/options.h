#pragma once

#include "aliases.h"
#include <spdlog/spdlog.h>
#include <string>

namespace bench_generalized {
    struct Options {
        int                       case_id = 1;
        Algo                      algo    = Algo::gdplusk;
        std::string               matrix_a_path;
        std::string               matrix_b_path;
        std::string               initial_guess;
        std::string               save_eigvec;
        std::string               save_results;
        int                       nev             = 1;     /*!< Number of requested eigenpairs. */
        int                       ncv             = 8;     /*!< Maximum search-space columns. */
        int                       block_size      = 1;     /*!< Number of vectors in each solver block. */
        int                       maxRetainBlocks = 1;     /*!< Retained restart blocks for lanczos. */
        int                       max_iters       = 100;   /*!< Maximum outer solver iterations; negative means unlimited. */
        int                       max_matvecs     = -1;    /*!< Maximum total matrix-vector products; negative means unlimited. */
        int                       inner_max_iters = 1000;  /*!< Maximum inner iterations for the inner correction solver. */
        int                       reps            = 1;     /*!< Number of benchmark repetitions. */
        double                    abstol          = 1e-12; /*!< Absolute residual-norm convergence tolerance, or rescaled residual tolerance when enabled. */
        double                    reltol          = 0.0;   /*!< Stabilized-reference residual reduction tolerance; zero disables it. */
        bool                      quit_when_saturated          = true; /*!< Stop after confirmed Ritz and residual saturation. */
        double                    inner_tol                    = 0.1;  /*!< Target residual reduction for the inner correction solver. */
        double                    ritz_saturation_tolerance    = 1e-5; /*!< Relative Ritz-value drift threshold per matvec. */
        int                       auto_probe_length            = 5;    /*!< Minimum outer iterations using the method tested by each AUTO probe. */
        int                       auto_max_probes              = 1; /*!< Maximum AUTO probes while Ritz values remain stabilized; -1 allows unlimited probes. */
        unsigned int              seed                         = 0;
        grit::Ritz                ritz                         = grit::Ritz::SR;
        spdlog::level::level_enum log_level                    = spdlog::level::warn;
        ResidualCorrection        residual_correction          = ResidualCorrection::NONE;
        bool                      use_refined_rayleigh_ritz    = false;
        bool                      use_b_inner_product          = false;
        bool                      use_jd_b_only                = false;
        bool                      use_rescaled_rnorm_tolerance = false; /*!< Interpret abstol as a derived rescaled residual-norm tolerance. */
        bool                      use_adaptive_inner_tolerance = false; /*!< Adapt inner_tol from previous inner-solver work. */
    };
}

#pragma once

#include "aliases.h"
#include <spdlog/spdlog.h>
#include <string>

namespace bench_standard {
    struct Options {
        int           case_id     = 1;
        std::string   matrix_path = "Please provide a path to a matrix in .mtx format";
        std::string   initial_guess;
        std::string   save_eigvec;
        std::string   save_results;
        int           nev                           = 1;     /*!< Number of requested eigenpairs. */
        int           ncv                           = 8;     /*!< Maximum search-space columns. */
        int           block_size                    = 1;     /*!< Number of vectors in each solver block. */
        int           max_basis_blocks              = 8;     /*!< Basis-block count used when ncv is derived. */
        int           max_iters                     = 100;   /*!< Maximum outer solver iterations; negative means unlimited. */
        int           max_matvecs                   = -1;    /*!< Maximum total matrix-vector products; negative means unlimited. */
        int           inner_max_iters               = 1000;  /*!< Maximum iterations for the inner correction solver. */
        int           reps                          = 1;     /*!< Number of benchmark repetitions. */
        double        tol                           = 1e-12; /*!< Absolute residual-norm convergence tolerance, or relative residual tolerance when enabled. */
        double        tol_rnorm_relative            = 0.0;   /*!< Relative-to-initial absolute residual-norm convergence tolerance; zero disables it. */
        double        sat_eigval_threshold          = 0.0;   /*!< Eigenvalue saturation threshold for stopping; zero disables this stop. */
        double        sat_rnorm_threshold           = 0.0;   /*!< Derived relative-residual saturation threshold for stopping; zero disables this stop. */
        double        inner_tol                     = 0.1;   /*!< Target residual reduction for the inner correction solver. */
        int           auto_min_dwell_iters          = 10;    /*!< Minimum consecutive cheap-Olsen AUTO steps before JD activation may occur. */
        double        auto_sat_eigval_threshold     = 1e-3;  /*!< Eigenvalue saturation threshold for AUTO JD activation. */
        double        auto_sat_rnorm_threshold      = 1e-2;  /*!< Derived relative-residual saturation threshold for AUTO JD activation. */
        double        auto_jd_start_rnorm_threshold = 1e-5;  /*!< Derived relative residual norm below which AUTO may activate JD; zero disables it. */
        int           auto_cheap_probe_interval     = 5;     /*!< JD steps before AUTO forces a cheap-Olsen probe. */
        double        auto_cheap_probe_factor = 1.0; /*!< Cheap probe must improve the Ritz value by this factor times max(absolute rnorm^2, roundoff scale). */
        unsigned int  seed                    = 0;
        grit::OptRitz ritz                    = grit::OptRitz::SR;
        spdlog::level::level_enum log_level   = spdlog::level::warn;
        ResidualCorrection        residual_correction          = ResidualCorrection::NONE;
        bool                      use_refined_rayleigh_ritz    = false;
        bool                      use_relative_rnorm_tolerance = false; /*!< Interpret tol as a derived relative residual-norm tolerance. */
        bool                      use_adaptive_inner_tolerance = false; /*!< Adapt inner_tol from previous inner-solver work. */
    };
}

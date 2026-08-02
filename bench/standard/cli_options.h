#pragma once

#include "aliases.h"
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

namespace bench_standard {
    struct CliOptions {
        bench_standard::Algo algo        = bench_standard::Algo::gdplusk;
        std::string          matrix_path = "Please provide a path to an matrix in .mtx format";
        std::string          initial_guess;
        std::string          save_eigvec;
        std::string          save_results;
        bool                 print_summary        = false;
        int                  nev                  = 1;       /*!< Number of requested eigenpairs. */
        std::vector<int>     ncv                  = {8};     /*!< Maximum search-space columns, optionally as a sweep list. */
        std::vector<int>     block_size           = {1};     /*!< Solver block size, optionally as a sweep list. */
        std::vector<int>     max_retain_blocks    = {1};     /*!< Retained restart blocks for lanczos, optionally as a sweep list. */
        int                  max_iters            = 100;     /*!< Maximum outer solver iterations; negative means unlimited. */
        int                  max_matvecs          = -1;      /*!< Maximum total matrix-vector products; negative means unlimited. */
        std::vector<int>     inner_max_iters      = {1000};  /*!< Maximum inner correction iterations, optionally as a sweep list. */
        int                  reps                 = 1;       /*!< Number of benchmark repetitions. */
        std::vector<double>  abstol               = {1e-12}; /*!< Absolute residual-norm convergence tolerance, or rescaled residual tolerance when enabled. */
        double               reltol               = 0.0;     /*!< Stabilized-reference residual reduction tolerance; zero disables it. */
        double               sat_eigval_threshold = 0.0;     /*!< Eigenvalue saturation threshold for stopping; zero disables this stop. */
        double               sat_rnorm_threshold  = 0.0;     /*!< Rescaled residual saturation threshold for stopping; zero disables this stop. */
        std::vector<double>  inner_tol            = {0.1};   /*!< Inner correction tolerance, optionally as a sweep list. */
        double               ritz_stabilization_tolerance = 1e-3; /*!< Residual-relative Ritz stabilization and AUTO probe progress tolerance. */
        int                  auto_probe_interval           = 5;    /*!< Active-method outer iterations between AUTO probes. */
        int                  auto_probe_length             = 3;    /*!< Outer iterations using the method tested by each AUTO probe. */
        unsigned int         seed                          = 0;
        std::string               ritz = "SR";
        spdlog::level::level_enum log_level                    = spdlog::level::warn;
        std::string               residual_correction          = "none";
        std::vector<bool>         use_refined_rayleigh_ritz    = {false};
        bool                      use_rescaled_rnorm_tolerance = false;   /*!< Interpret abstol as a derived rescaled residual-norm tolerance. */
        std::vector<bool>         use_adaptive_inner_tolerance = {false}; /*!< Adapt inner_tol from previous inner-solver work. */

        bool explicit_inner_max_iters               = false;
        bool explicit_max_retain_blocks             = false;
        bool explicit_inner_tol                     = false;
        bool explicit_residual_correction           = false;
        bool explicit_auto_probe_interval            = false;
        bool explicit_auto_probe_length              = false;
        bool explicit_use_adaptive_inner_tolerance  = false;
    };
}

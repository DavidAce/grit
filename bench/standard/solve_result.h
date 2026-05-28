#pragma once

#include "aliases.h"
#include <cstdint>
#include <h5pp/h5pp.h>
#include <limits>
#include <vector>

namespace bench_standard {
    struct SolverSnapshot {
        using vlen_type = h5pp::varr_t<int64_t>;

        int32_t               case_id  = 0;
        int32_t               rep      = 0;
        uint32_t              seed     = 0;
        uint32_t              rep_seed = 0;
        h5pp::fstr_t<1024>    matrix_path;
        h5pp::fstr_t<1024>    initial_guess;
        int32_t               nev                           = 0;
        int32_t               ncv                           = 0;
        int32_t               block_size                    = 0;
        int32_t               max_basis_blocks              = 0;
        int32_t               max_iters                     = 0;
        int32_t               max_matvecs                   = 0;
        int32_t               inner_max_iters               = 0;
        double                tol                           = 0.0;
        double                tol_rnorm_relative            = 0.0;
        double                sat_eigval_threshold          = 0.0;
        double                sat_rnorm_threshold           = 0.0;
        double                inner_tol                     = 0.0;
        int32_t               auto_min_dwell_iters          = 0;
        double                auto_sat_eigval_threshold     = 0.0;
        double                auto_sat_rnorm_threshold      = 0.0;
        double                auto_jd_start_rnorm_threshold = 0.0;
        int32_t               auto_cheap_probe_interval     = 0;
        double                auto_cheap_probe_factor       = 0.0;
        h5pp::fstr_t<16>      ritz;
        h5pp::fstr_t<32>      residual_correction;
        uint8_t               use_refined_rayleigh_ritz    = 0;
        uint8_t               use_relative_rnorm_tolerance = 0;
        uint8_t               use_adaptive_inner_tolerance = 0;
        h5pp::fstr_t<64>      stop_reason;
        double                eigenvalue             = 0.0;
        double                rnorm                  = 0.0;
        double                rrnorm                 = 0.0;
        int64_t               iterations             = 0;
        int64_t               matvecs                = 0;
        int64_t               outer_matvecs          = 0;
        int64_t               inner_matvecs          = 0;
        int64_t               inner_iterations       = 0;
        int64_t               jdops_inner            = 0;
        int64_t               precond                = 0;
        int64_t               precond_inner          = 0;
        int64_t               precond_total          = 0;
        double                inner_tol_last         = 0.0;
        double                inner_error_last       = 0.0;
        int64_t               first_cheap_to_jd_iter = -1;
        h5pp::fstr_t<32>      residual_correction_active;
        h5pp::fstr_t<32>      residual_correction_step;
        int64_t               auto_dwell                = 0;
        int64_t               auto_jd_steps_since_probe = 0;
        int64_t               saturation_count_eigval   = 0;
        int64_t               saturation_count_rnorm    = 0;
        int64_t               saturation_count_max      = 0;
        double                op_norm_estimate          = 0.0;
        double                condition                 = 0.0;
        double                sensitivity               = 0.0;
        double                gap                       = std::numeric_limits<double>::infinity();
        uint8_t               rnorm_below_tol           = 0;
        uint8_t               rnorm_below_gap           = 0;
        int64_t               num_cheap_to_jd_switches  = 0;
        int64_t               num_jd_to_cheap_switches  = 0;
        h5pp::varr_t<int64_t> cheap_to_jd_switch_iters;
        h5pp::varr_t<int64_t> jd_to_cheap_switch_iters;
        double                seconds    = 0.0;
        double                vmrss_mib  = -1.0;
        double                vmhwm_mib  = -1.0;
        double                vmpeak_mib = -1.0;
    };

    struct SolveResult {
        SolverSnapshot              final;
        std::vector<SolverSnapshot> snapshots;
        DenseMatrix                 eigvecs;
    };
}

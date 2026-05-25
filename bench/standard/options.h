#pragma once

#include "aliases.h"

#include <spdlog/spdlog.h>
#include <string>

namespace bench_standard {
    struct Options {
        int                       case_id                  = 1;
        std::string               matrix                   = std::string(GRIT_BENCH_DATA_DIR) + "/finance256/finance256.mtx";
        std::string               initial_guess;
        std::string               save_eigvec;
        int                       nev                      = 1;
        int                       ncv                      = 8;
        int                       block_size               = 1;
        int                       max_basis_blocks         = 8;
        int                       max_iters                = 100;
        int                       max_matvecs              = -1;
        int                       inner_iters              = 1000;
        int                       reps                     = 1;
        double                    tol                      = 1e-12;
        double                    reltol                   = 0.0;
        double                    tol_stall_evals  = 0.0;
        double                    tol_stall_rnorm = 0.0;
        double                    inner_tol                = 0.1;
        unsigned int              seed                     = 0;
        grit::OptRitz             ritz                     = grit::OptRitz::SR;
        spdlog::level::level_enum log_level                = spdlog::level::warn;
        ResidualCorrection        residual_correction      = ResidualCorrection::NONE;
        bool                      refined_rayleigh_ritz    = false;
        bool                      relative_rnorm_tolerance = false;
        bool                      adaptive_inner_tolerance = true;
    };
}

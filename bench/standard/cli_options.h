#pragma once

#include <spdlog/spdlog.h>
#include <string>

namespace bench_standard {
    struct CliOptions {
        std::string               matrix                   = std::string(GRIT_BENCH_DATA_DIR) + "/finance256/finance256.mtx";
        std::string               initial_guess;
        std::string               save_eigvec;
        int                       nev                      = 1;
        std::string               ncv                      = "8";
        std::string               block_size               = "1";
        int                       max_basis_blocks         = 8;
        int                       max_iters                = 100;
        int                       max_matvecs              = -1;
        std::string               inner_iters              = "1000";
        int                       reps                     = 1;
        std::string               tol                      = "1e-12";
        double                    reltol                   = 0.0;
        double                    tol_stall_evals  = 0.0;
        double                    tol_stall_rnorm = 0.0;
        std::string               inner_tol                = "0.1";
        unsigned int              seed                     = 0;
        std::string               ritz                     = "sr";
        spdlog::level::level_enum log_level                = spdlog::level::warn;
        std::string               residual_correction      = "none";
        std::string               refined_rayleigh_ritz    = "false";
        bool                      relative_rnorm_tolerance = false;
        std::string               adaptive_inner_tolerance = "true";
    };
}

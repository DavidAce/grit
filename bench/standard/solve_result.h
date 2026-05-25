#pragma once

#include "aliases.h"

#include <string>

namespace bench_standard {
    struct SolveResult {
        int                case_id                  = 1;
        int                rep                      = 0;
        int                ncv                      = 0;
        int                block_size               = 0;
        int                inner_iters              = 0;
        double             tol                      = 0.0;
        double             inner_tol                = 0.0;
        grit::OptRitz      ritz                     = grit::OptRitz::SR;
        ResidualCorrection residual_correction      = ResidualCorrection::NONE;
        bool               refined_rayleigh_ritz    = false;
        bool               adaptive_inner_tolerance = true;
        std::string        stop_reason;
        double             eigenvalue               = 0.0;
        double             residual                 = 0.0;
        Eigen::Index       iterations               = 0;
        Eigen::Index       matvecs                  = 0;
        Eigen::Index       outer_matvecs            = 0;
        Eigen::Index       inner_matvecs            = 0;
        Eigen::Index       jdops_inner              = 0;
        double             seconds                  = 0.0;
        double             vmrss_mib                = -1.0;
        double             vmhwm_mib                = -1.0;
        double             vmpeak_mib               = -1.0;
        DenseMatrix        eigvecs;
    };
}

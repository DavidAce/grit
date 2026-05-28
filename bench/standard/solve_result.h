#pragma once

#include "aliases.h"
#include "options.h"
#include <string>
#include <vector>

namespace bench_standard {
    struct SolveResult {
        Options                   options;
        int                       rep                          = 0;
        unsigned int              rep_seed                     = 0;
        int                       inner_iters                  = 0;
        double                    inner_tol                    = 0.0;
        std::string               stop_reason;
        double                    eigenvalue             = 0.0;
        double                    residual               = 0.0;
        Eigen::Index              iterations             = 0;
        Eigen::Index              matvecs                = 0;
        Eigen::Index              outer_matvecs          = 0;
        Eigen::Index              inner_matvecs          = 0;
        Eigen::Index              jdops_inner            = 0;
        Eigen::Index              first_cheap_to_jd_iter = -1;
        std::vector<Eigen::Index> cheap_to_jd_switch_iters;
        std::vector<Eigen::Index> jd_to_cheap_switch_iters;
        double                    seconds    = 0.0;
        double                    vmrss_mib  = -1.0;
        double                    vmhwm_mib  = -1.0;
        double                    vmpeak_mib = -1.0;
        DenseMatrix               eigvecs;
    };
}

#pragma once

#include <string_view>

namespace grit {
    enum class OptRitz { NONE, LR, LM, SR, SM };

    enum class StopReason {
        none,
        converged,
        max_iters,
        max_matvecs,
        ritz_value_stalled,
        ritz_residual_stalled,
        lanczos_beta_stalled,
        subspace_exhausted,
        invalid_input
    };

    inline std::string_view enum2sv(OptRitz ritz) {
        switch(ritz) {
            case OptRitz::NONE: return "NONE";
            case OptRitz::LR: return "LR";
            case OptRitz::LM: return "LM";
            case OptRitz::SR: return "SR";
            case OptRitz::SM: return "SM";
        }
        return "NONE";
    }

    inline std::string_view enum2sv(StopReason reason) {
        switch(reason) {
            case StopReason::none: return "none";
            case StopReason::converged: return "converged";
            case StopReason::max_iters: return "max_iters";
            case StopReason::max_matvecs: return "max_matvecs";
            case StopReason::ritz_value_stalled: return "ritz_value_stalled";
            case StopReason::ritz_residual_stalled: return "ritz_residual_stalled";
            case StopReason::lanczos_beta_stalled: return "lanczos_beta_stalled";
            case StopReason::subspace_exhausted: return "subspace_exhausted";
            case StopReason::invalid_input: return "invalid_input";
        }
        return "none";
    }
}

#pragma once

#include <string>
#include <string_view>
#include <type_traits>

namespace grit {
    enum class OptRitz { NONE, LR, LM, SR, SM };

    enum class StopReason : int {
        none                   = 0,
        converged              = 1,
        ritz_residual_stalled  = 2,
        subspace_exhausted     = 4,
        ritz_value_stalled     = 16,
        max_iters              = 32,
        max_matvecs            = 64,
        lanczos_beta_stalled   = 128,
        invalid_input          = 256,
        allow_bitops
    };

    constexpr auto operator|(StopReason lhs, StopReason rhs) noexcept -> StopReason {
        using U = std::underlying_type_t<StopReason>;
        return static_cast<StopReason>(static_cast<U>(lhs) | static_cast<U>(rhs));
    }

    constexpr auto operator&(StopReason lhs, StopReason rhs) noexcept -> StopReason {
        using U = std::underlying_type_t<StopReason>;
        return static_cast<StopReason>(static_cast<U>(lhs) & static_cast<U>(rhs));
    }

    constexpr auto operator|=(StopReason &lhs, StopReason rhs) noexcept -> StopReason {
        lhs = lhs | rhs;
        return lhs;
    }

    inline bool has_flag(StopReason target, StopReason check) noexcept {
        using U = std::underlying_type_t<StopReason>;
        return (static_cast<U>(target) & static_cast<U>(check)) == static_cast<U>(check);
    }

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
            case StopReason::allow_bitops: return "allow_bitops";
        }
        return "multiple";
    }

    inline std::string enum2s(StopReason reason) {
        if(reason == StopReason::none) return std::string(enum2sv(reason));

        std::string msg;
        auto        append = [&](StopReason flag) {
            if(!has_flag(reason, flag)) return;
            if(!msg.empty()) msg += "|";
            msg += enum2sv(flag);
        };

        append(StopReason::converged);
        append(StopReason::max_iters);
        append(StopReason::max_matvecs);
        append(StopReason::ritz_value_stalled);
        append(StopReason::ritz_residual_stalled);
        append(StopReason::lanczos_beta_stalled);
        append(StopReason::subspace_exhausted);
        append(StopReason::invalid_input);
        if(msg.empty()) msg = "multiple";
        return msg;
    }
}

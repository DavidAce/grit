#pragma once

#include <string>
#include <string_view>
#include <type_traits>

namespace grit {
    /*! Eigenvalue problem type. */
    enum class Form {
        STANDARD,   /*!< Solve A x = lambda x. */
        GENERALIZED /*!< Solve A x = lambda B x. */
    };

    /*! Which part of the Ritz spectrum to select. */
    enum class Ritz {
        NONE, /*!< Do not select Ritz values. */
        LR,   /*!< Largest real part. */
        LM,   /*!< Largest magnitude. */
        SR,   /*!< Smallest real part. */
        SM    /*!< Smallest magnitude. */
    };

    /*! Residual correction used to expand the search space. */
    enum class ResidualCorrectionType {
        NONE,            /*!< Use the raw residual as correction. */
        CHEAP_OLSEN,     /*!< Use the cheap Olsen correction. */
        FULL_OLSEN,      /*!< Use the full Olsen correction. */
        JACOBI_DAVIDSON, /*!< Use a Jacobi-Davidson correction equation. */
        AUTO             /*!< Let the solver switch between cheap Olsen and Jacobi-Davidson. */
    };

    /*! Reason why an eigensolver stopped. */
    enum class StopReason : int {
        none                  = 0,   /*!< Solver has not stopped. */
        converged             = 1,   /*!< Requested Ritz pairs reached the residual tolerance. */
        ritz_residual_stalled = 2,   /*!< Ritz residuals stopped changing enough to continue. */
        subspace_exhausted    = 4,   /*!< No useful new search-space vector could be added. */
        ritz_value_stalled    = 16,  /*!< Ritz values stopped changing enough to continue. */
        max_iters             = 32,  /*!< Maximum outer iterations reached. */
        max_matvecs           = 64,  /*!< Maximum matrix-vector products reached. */
        lanczos_beta_stalled  = 128, /*!< Lanczos recurrence beta became too small. */
        invalid_input         = 256, /*!< Configuration or input operators are invalid. */
        allow_bitops
    };

    /*!
     * Combine two stop reasons.
     * @param lhs First stop reason.
     * @param rhs Second stop reason.
     * @return Combined stop reason.
     */
    constexpr auto operator|(StopReason lhs, StopReason rhs) noexcept -> StopReason {
        using U = std::underlying_type_t<StopReason>;
        return static_cast<StopReason>(static_cast<U>(lhs) | static_cast<U>(rhs));
    }

    /*!
     * Intersect two stop reasons.
     * @param lhs First stop reason.
     * @param rhs Second stop reason.
     * @return Common stop reason flags.
     */
    constexpr auto operator&(StopReason lhs, StopReason rhs) noexcept -> StopReason {
        using U = std::underlying_type_t<StopReason>;
        return static_cast<StopReason>(static_cast<U>(lhs) & static_cast<U>(rhs));
    }

    /*!
     * Add a stop reason flag in place.
     * @param lhs Stop reason updated in place.
     * @param rhs Stop reason flag to add.
     * @return Updated stop reason.
     */
    constexpr auto operator|=(StopReason &lhs, StopReason rhs) noexcept -> StopReason {
        lhs = lhs | rhs;
        return lhs;
    }

    /*!
     * Check whether a stop reason contains a given flag.
     * @param target Stop reason to inspect.
     * @param check Stop reason flag to check.
     * @return True when target contains check.
     */
    inline bool has_flag(StopReason target, StopReason check) noexcept {
        using U = std::underlying_type_t<StopReason>;
        return (static_cast<U>(target) & static_cast<U>(check)) == static_cast<U>(check);
    }

    /*!
     * Return the short name of a Ritz selector.
     * @param ritz Ritz selector.
     * @return Short name.
     */
    inline std::string_view enum2sv(Ritz ritz) {
        switch(ritz) {
            case Ritz::NONE: return "NONE";
            case Ritz::LR: return "LR";
            case Ritz::LM: return "LM";
            case Ritz::SR: return "SR";
            case Ritz::SM: return "SM";
        }
        return "NONE";
    }

    /*! Return the short name of a residual correction selector. */
    inline std::string_view enum2sv(ResidualCorrectionType correction) {
        switch(correction) {
            case ResidualCorrectionType::NONE: return "NONE";
            case ResidualCorrectionType::CHEAP_OLSEN: return "CHEAP_OLSEN";
            case ResidualCorrectionType::FULL_OLSEN: return "FULL_OLSEN";
            case ResidualCorrectionType::JACOBI_DAVIDSON: return "JACOBI_DAVIDSON";
            case ResidualCorrectionType::AUTO: return "AUTO";
        }
        return "NONE";
    }

    /*!
     * Return the short name of a single stop reason.
     * @param reason Stop reason.
     * @return Short name.
     */
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

    /*!
     * Return the short name of a possibly combined stop reason.
     * @param reason Stop reason.
     * @return Short name.
     */
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

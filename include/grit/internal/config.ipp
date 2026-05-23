#pragma once

#include <grit/config.h>

namespace grit {
    template<typename Scalar_>
    gdplusk_config<Scalar_> &gdplusk_config<Scalar_>::set_initial_guess(const MatrixType &V) {
        initial_guess_ptr = &V;
        return *this;
    }

    template<typename Scalar_>
    gdplusk_config<Scalar_> &gdplusk_config<Scalar_>::clear_initial_guess() {
        initial_guess_ptr = nullptr;
        return *this;
    }

    template<typename Scalar_>
    bool gdplusk_config<Scalar_>::has_initial_guess() const {
        return initial_guess_ptr != nullptr;
    }

    template<typename Scalar_>
    const typename gdplusk_config<Scalar_>::MatrixType &gdplusk_config<Scalar_>::initial_guess() const {
        if(initial_guess_ptr) return *initial_guess_ptr;
        return empty_initial_guess;
    }
}

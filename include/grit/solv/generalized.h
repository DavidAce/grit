#pragma once

#include "../algo/gdplusk.h"
#include "../algo/lanczos.h"
#include "../algo/lobpcg.h"

namespace grit::generalized {
    /*! GD+K solver for generalized eigenvalue problems. */
    template<typename Scalar>
    using gdplusk = algo::gdplusk<Scalar, grit::Form::GENERALIZED>;

    /*! Lanczos solver for generalized eigenvalue problems. */
    template<typename Scalar>
    using lanczos = algo::lanczos<Scalar, grit::Form::GENERALIZED>;

    /*! LOBPCG solver for generalized eigenvalue problems. */
    template<typename Scalar>
    using lobpcg = algo::lobpcg<Scalar, grit::Form::GENERALIZED>;
}

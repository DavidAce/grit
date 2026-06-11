#pragma once

#include "../algo/gdplusk.h"
#include "../algo/lanczos.h"
#include "../algo/lobpcg.h"

namespace grit::standard {
    /*! GD+K solver for standard eigenvalue problems. */
    template<typename Scalar>
    using gdplusk = algo::gdplusk<Scalar, grit::Form::STANDARD>;

    /*! Lanczos solver for standard eigenvalue problems. */
    template<typename Scalar>
    using lanczos = algo::lanczos<Scalar, grit::Form::STANDARD>;

    /*! LOBPCG solver for standard eigenvalue problems. */
    template<typename Scalar>
    using lobpcg = algo::lobpcg<Scalar, grit::Form::STANDARD>;
}

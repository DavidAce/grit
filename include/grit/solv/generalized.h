#pragma once

#include "../algo/gdplusk.h"
#include "../algo/lanczos.h"
#include "../algo/lobpcg.h"

namespace grit::generalized {
    template<typename Scalar>
    using gdplusk = algo::gdplusk<Scalar, grit::Form::GENERALIZED>;

    template<typename Scalar>
    using lanczos = algo::lanczos<Scalar, grit::Form::GENERALIZED>;

    template<typename Scalar>
    using lobpcg = algo::lobpcg<Scalar, grit::Form::GENERALIZED>;
}

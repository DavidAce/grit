#pragma once

#include "../algo/gdplusk.h"
#include "../algo/lanczos.h"
#include "../algo/lobpcg.h"
#include "../form/generalized.h"
#include "../prob/generalized.h"

namespace grit::generalized {
    template<typename Scalar>
    using gdplusk = algo::gdplusk<form::generalized<Scalar>>;

    template<typename Scalar>
    using lanczos = algo::lanczos<form::generalized<Scalar>>;

    template<typename Scalar>
    using lobpcg = algo::lobpcg<form::generalized<Scalar>>;
}

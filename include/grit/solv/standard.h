#pragma once

#include "../algo/gdplusk.h"
#include "../algo/lanczos.h"
#include "../algo/lobpcg.h"
#include "../form/standard.h"
#include "../prob/standard.h"

namespace grit::standard {
    template<typename Scalar>
    using gdplusk = algo::gdplusk<form::standard<Scalar>>;

    template<typename Scalar>
    using lanczos = algo::lanczos<form::standard<Scalar>>;

    template<typename Scalar>
    using lobpcg = algo::lobpcg<form::standard<Scalar>>;
}

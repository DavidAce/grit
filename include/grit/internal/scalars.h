#pragma once

#include <complex>
#include <type_traits>

using fp32 = float;
using fp64 = double;
using fp80 = long double;

using cx32 = std::complex<fp32>;
using cx64 = std::complex<fp64>;
using cx80 = std::complex<fp80>;

namespace grit {
    struct type_unavailable {
        type_unavailable() = delete;
    };
}

#if defined(GRIT_USE_FLOAT128)
    #include <stdfloat>
using fp128 = std::float128_t;
using cx128 = std::complex<fp128>;
#else
using fp128 = grit::type_unavailable;
using cx128 = std::complex<fp128>;
#endif

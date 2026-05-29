#pragma once

#include "gdplusk.h"

namespace grit::algo {
    template<typename Scalar, grit::Form form>
    class lobpcg : public gdplusk<Scalar, form> {
        public:
        using gdplusk<Scalar, form>::gdplusk;
    };
}

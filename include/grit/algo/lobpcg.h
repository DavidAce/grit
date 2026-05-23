#pragma once

#include "gdplusk.h"

namespace grit::algo {
    template<typename Form>
    class lobpcg : public gdplusk<Form> {
        public:
        using gdplusk<Form>::gdplusk;
    };
}

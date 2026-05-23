#pragma once

#include "gdplusk.h"

namespace grit::algo {
    template<typename Form>
    class lanczos : public gdplusk<Form> {
        public:
        using gdplusk<Form>::gdplusk;
    };
}

#pragma once

#include "base.h"

namespace grit::form {
    /*! Base form for standard eigenvalue problems A x = lambda x. */
    template<typename Scalar>
    using standard = base<Scalar, grit::Form::STANDARD>;
}

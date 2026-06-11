#pragma once

#include "base.h"

namespace grit::form {
    /*! Base form for generalized eigenvalue problems A x = lambda B x. */
    template<typename Scalar>
    using generalized = base<Scalar, grit::Form::GENERALIZED>;
}

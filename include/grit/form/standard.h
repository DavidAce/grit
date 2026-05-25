#pragma once

#include "base.h"

namespace grit::form {
    template<typename Scalar_>
    class standard : public base<Scalar_> {
        public:
        using base<Scalar_>::base;
        using Scalar     = typename base<Scalar_>::Scalar;
        using MatrixType = typename base<Scalar_>::MatrixType;
        using base<Scalar_>::Q;
        using base<Scalar_>::AQ;
        using base<Scalar_>::BQ;
        using base<Scalar_>::T1;
        using base<Scalar_>::T2;
        using base<Scalar_>::T_evals;
        using base<Scalar_>::T_evecs;
        using base<Scalar_>::half;
        using base<Scalar_>::status;

        std::string_view form_name() const final;
        bool             is_generalized_problem() const final;
        MatrixType       MultB(const Eigen::Ref<const MatrixType> &X) final;
        MatrixType       MultB_inner(const Eigen::Ref<const MatrixType> &X) final;
        void             diagonalizeT() final;
    };
}

#pragma once

#include "base.h"

namespace grit::form {
    template<typename Scalar_>
    class generalized : public base<Scalar_> {
        public:
        using Scalar     = typename base<Scalar_>::Scalar;
        using RealScalar = typename base<Scalar_>::RealScalar;
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

        generalized(Eigen::Index nev, Eigen::Index ncv, OptRitz ritz, const MatrixType &V, MatVec<Scalar> &A, MatVec<Scalar> &B,
                    spdlog::level::level_enum logLevel_ = spdlog::level::warn);

        std::string_view form_name() const final;
        MatrixType       MultB(const Eigen::Ref<const MatrixType> &X) final;
        void             diagonalizeT() final;

        MatVec<Scalar> &B;

        protected:
        void set_form_jcbMaxBlockSize(Eigen::Index jcbMaxBlockSize) final;
        void set_form_jcbOverlapSize(Eigen::Index jcbOverlapSize) final;
        void set_form_jcbNumPasses(Eigen::Index jcbNumPasses) final;
        void set_form_preconditioner_type(Preconditioner preconditioner_type) final;
        void set_form_preconditioner_params(Eigen::Index maxiters, RealScalar initialTol, Eigen::Index jcbMaxBlockSize) final;
    };
}

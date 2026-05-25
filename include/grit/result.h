#pragma once

#include "enums.h"
#include <complex>
#include <Eigen/Core>
#include <utility>

namespace grit::form {
    template<typename Scalar>
    class base;
}

namespace grit {
    template<typename Scalar_>
    class result_view {
        public:
        using Scalar     = Scalar_;
        using RealScalar = decltype(std::real(std::declval<Scalar>()));
        using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorReal = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;

        result_view() = default;

        [[nodiscard]] const VectorReal &eigVal() const;
        [[nodiscard]] const MatrixType &eigVecs() const;
        [[nodiscard]] const VectorReal &rNorms() const;
        [[nodiscard]] StopReason        stopReason() const;
        [[nodiscard]] Eigen::Index      iter() const;
        [[nodiscard]] Eigen::Index      num_matvecs() const;
        [[nodiscard]] Eigen::Index      num_matvecs_inner() const;
        [[nodiscard]] Eigen::Index      num_jdops_inner() const;
        [[nodiscard]] Eigen::Index      num_matvecs_total() const;
        [[nodiscard]] Eigen::Index      num_precond_total() const;

        private:
        friend class form::base<Scalar>;

        explicit result_view(const form::base<Scalar> &source_);

        const form::base<Scalar> *source = nullptr;
    };
}

#pragma once

#include "../MatVec.h"
#include <optional>

namespace grit::generalized {
    template<typename Scalar_>
    class problem {
        public:
        using Scalar     = Scalar_;
        using RealScalar = typename MatVec<Scalar>::RealScalar;

        problem() = default;
        problem(MatVec<Scalar> &A_, MatVec<Scalar> &B_);

        void set_A(MatVec<Scalar> &A_);
        void set_B(MatVec<Scalar> &B_);

        [[nodiscard]] bool has_A() const;
        [[nodiscard]] bool has_B() const;

        [[nodiscard]] MatVec<Scalar> &get_A() const;
        [[nodiscard]] MatVec<Scalar> &get_B() const;

        void                       set_size(Eigen::Index size);
        [[nodiscard]] Eigen::Index get_size() const;

        void set_A_op_norm(RealScalar op_norm);
        void set_B_op_norm(RealScalar op_norm);

        [[nodiscard]] std::optional<RealScalar> get_A_op_norm() const;
        [[nodiscard]] std::optional<RealScalar> get_B_op_norm() const;

        private:
        MatVec<Scalar>             *A = nullptr;
        MatVec<Scalar>             *B = nullptr;
        std::optional<Eigen::Index> size_;
        std::optional<RealScalar>   A_op_norm;
        std::optional<RealScalar>   B_op_norm;
    };
}

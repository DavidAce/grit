#pragma once

#include "../MatVec.h"
#include <optional>

namespace grit::standard {
    template<typename Scalar_>
    class problem {
        public:
        using Scalar     = Scalar_;
        using RealScalar = typename MatVec<Scalar>::RealScalar;

        problem() = default;
        explicit problem(MatVec<Scalar> &A_);

        void set_A(MatVec<Scalar> &A_);

        [[nodiscard]] bool            has_A() const;
        [[nodiscard]] MatVec<Scalar> &get_A() const;

        void                       set_size(Eigen::Index size);
        [[nodiscard]] Eigen::Index get_size() const;

        void                                    set_op_norm(RealScalar op_norm);
        [[nodiscard]] std::optional<RealScalar> get_op_norm() const;

        private:
        MatVec<Scalar>             *A = nullptr;
        std::optional<Eigen::Index> size_;
        std::optional<RealScalar>   A_op_norm;
    };
}

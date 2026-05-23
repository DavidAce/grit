#pragma once

#include <chrono>
#include <Eigen/Core>
#include <functional>

namespace grit::internal::precondition {
    template<typename Scalar_>
    class MatrixLikeOperator;
}

template<typename T>
using GritDenseMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

namespace Eigen::internal {
    template<typename Scalar_>
    struct traits<grit::internal::precondition::MatrixLikeOperator<Scalar_>> : public Eigen::internal::traits<GritDenseMatrix<Scalar_>> {};
}

namespace grit::internal::precondition {
    template<typename Scalar_>
    class MatrixLikeOperator : public Eigen::EigenBase<MatrixLikeOperator<Scalar_>> {
        public:
        using Scalar     = Scalar_;
        using RealScalar = decltype(std::real(std::declval<Scalar>()));
        using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        typedef int StorageIndex;
        enum { ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic, IsRowMajor = false };

        static constexpr bool has_projector_op = false;
        mutable Eigen::Index  m_opcounter      = 0;
        mutable double        m_optimer        = 0.0;
        Eigen::Index          size             = 0;

        MatrixLikeOperator() = default;
        MatrixLikeOperator(Eigen::Index size_, std::function<MatrixType(const Eigen::Ref<const MatrixType> &)> MatrixOp_)
            : size(size_), MatrixOp(std::move(MatrixOp_)) {}
        std::function<MatrixType(const Eigen::Ref<const MatrixType> &)> MatrixOp;

        template<typename Rhs>
        Eigen::Product<MatrixLikeOperator, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs> &x) const {
            auto t_start  = std::chrono::high_resolution_clock::now();
            auto result   = Eigen::Product<MatrixLikeOperator, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
            auto t_end    = std::chrono::high_resolution_clock::now();
            m_optimer    += std::chrono::duration<double>(t_end - t_start).count();
            m_opcounter++;
            return result;
        }

        template<typename Rhs>
        Eigen::Product<MatrixLikeOperator, Rhs, Eigen::AliasFreeProduct> gemm(const Eigen::MatrixBase<Rhs> &x) const {
            return operator*(x);
        }

        [[nodiscard]] Eigen::Index rows() const { return size; }
        [[nodiscard]] Eigen::Index cols() const { return size; }
        Eigen::Index               iterations() const { return m_opcounter; }
        double                     elapsed_time() const { return m_optimer; }
    };
}

namespace Eigen::internal {
    template<typename Rhs, typename ReplScalar>
    struct generic_product_impl<grit::internal::precondition::MatrixLikeOperator<ReplScalar>, Rhs, DenseShape, DenseShape, GemvProduct>
        : generic_product_impl_base<grit::internal::precondition::MatrixLikeOperator<ReplScalar>, Rhs,
                                    generic_product_impl<grit::internal::precondition::MatrixLikeOperator<ReplScalar>, Rhs>> {
        typedef typename Product<grit::internal::precondition::MatrixLikeOperator<ReplScalar>, Rhs>::Scalar Scalar;

        template<typename Dest>
        static void scaleAndAddTo(Dest &dst, const grit::internal::precondition::MatrixLikeOperator<ReplScalar> &mat, const Rhs &rhs, const Scalar &alpha) {
            assert(alpha == Scalar(1) && "scaling is not implemented");
            EIGEN_ONLY_USED_FOR_DEBUG(alpha);
            dst.noalias() += mat.MatrixOp(rhs);
        }
    };
}

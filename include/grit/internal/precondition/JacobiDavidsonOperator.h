#pragma once

#include "../eigen/BiCGSTAB_ST.h"
#include "../eigen/ConjugateGradient_ST.h"
#include "../eigen/MINRES_ST.h"
#include "IterativeLinearSolverConfig.h"
#include "IterativeLinearSolverPreconditioner.h"
#include <chrono>
#include <complex>
#include <Eigen/Core>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace grit::internal::precondition {
    template<typename Scalar_>
    class JacobiDavidsonOperator;

    template<typename T>
    using DenseMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
}

namespace Eigen::internal {
    template<typename Scalar_>
    struct traits<grit::internal::precondition::JacobiDavidsonOperator<Scalar_>>
        : public Eigen::internal::traits<grit::internal::precondition::DenseMatrix<Scalar_>> {};
}

namespace grit::internal::precondition {
    template<typename Scalar_>
    class JacobiDavidsonOperator : public Eigen::EigenBase<JacobiDavidsonOperator<Scalar_>> {
        public:
        using Scalar     = Scalar_;
        using RealScalar = decltype(std::real(std::declval<Scalar>()));
        using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        typedef int StorageIndex;
        enum { ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic, IsRowMajor = false };

        mutable Eigen::Index m_opcounter = 0;
        mutable double       m_optimer   = 0.0;

        static constexpr auto eps = std::numeric_limits<RealScalar>::epsilon();
        mutable VectorType    x_tmp;
        mutable VectorType    y_tmp;
        mutable VectorType    z_tmp;
        const Eigen::Index    size             = 0;
        static constexpr bool has_projector_op = true;

        public:
        std::function<void(const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> Y)> ResidualOp;
        std::function<void(const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> Y)> ProjectOpL;
        std::function<void(const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> Y)> ProjectOpR;
        std::function<MatrixType(const Eigen::Ref<const MatrixType> &)>                      MatrixOp;

        void _check_template_params() {};
        JacobiDavidsonOperator() = default;
        JacobiDavidsonOperator(Eigen::Index size_, std::function<void(const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> Y)> ResidualOp_,
                               std::function<void(const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> Y)> ProjectOpL_,
                               std::function<void(const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> Y)> ProjectOpR_,
                               std::function<MatrixType(const Eigen::Ref<const MatrixType> &)>                      MatrixOp_)
            : size(size_), ResidualOp(std::move(ResidualOp_)), ProjectOpL(std::move(ProjectOpL_)), ProjectOpR(std::move(ProjectOpR_)),
              MatrixOp(std::move(MatrixOp_)) {}

        template<typename Rhs>
        Eigen::Product<JacobiDavidsonOperator, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs> &x) const {
            auto result = Eigen::Product<JacobiDavidsonOperator, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
            return result;
        }
        template<typename Rhs>
        Eigen::Product<JacobiDavidsonOperator, Rhs, Eigen::AliasFreeProduct> gemm(const Eigen::MatrixBase<Rhs> &x) {
            auto t_start  = std::chrono::high_resolution_clock::now();
            auto result   = MatrixOp(x);
            auto t_end    = std::chrono::high_resolution_clock::now();
            m_optimer    += std::chrono::duration<double>(t_end - t_start).count();
            m_opcounter++;
            return result;
        }
        [[nodiscard]] Eigen::Index rows() const { return size; };
        [[nodiscard]] Eigen::Index cols() const { return size; };
        Eigen::Index               iterations() const { return m_opcounter; }
        double                     elapsed_time() const { return m_optimer; }
    };

    template<typename T>
    struct is_std_complex : std::false_type {};
    template<typename T>
    struct is_std_complex<std::complex<T>> : std::true_type {};
}

namespace Eigen::internal {

    template<typename Rhs, typename ReplScalar>
    struct generic_product_impl<grit::internal::precondition::JacobiDavidsonOperator<ReplScalar>, Rhs, DenseShape, DenseShape, GemvProduct>
        : generic_product_impl_base<grit::internal::precondition::JacobiDavidsonOperator<ReplScalar>, Rhs,
                                    generic_product_impl<grit::internal::precondition::JacobiDavidsonOperator<ReplScalar>, Rhs>> {
        typedef typename Product<grit::internal::precondition::JacobiDavidsonOperator<ReplScalar>, Rhs>::Scalar Scalar;

        template<typename Dest>
        static void scaleAndAddTo(Dest &dst, const grit::internal::precondition::JacobiDavidsonOperator<ReplScalar> &mat, const Rhs &rhs, const Scalar &alpha) {
            assert(alpha == Scalar(1) && "scaling is not implemented");
            assert(rhs.size() == mat.rows());
            assert(dst.size() == mat.rows());
            EIGEN_ONLY_USED_FOR_DEBUG(alpha);
            {
                const auto &ResidualOp = mat.ResidualOp;
                const auto &ProjectOpL = mat.ProjectOpL;
                const auto &ProjectOpR = mat.ProjectOpR;
                auto       &x_tmp      = mat.x_tmp;
                auto       &y_tmp      = mat.y_tmp;
                auto       &z_tmp      = mat.z_tmp;

                x_tmp.resize(rhs.rows(), rhs.cols());
                y_tmp.resize(rhs.rows(), rhs.cols());
                z_tmp.resize(rhs.rows(), rhs.cols());

                auto t_start = std::chrono::high_resolution_clock::now();
                ProjectOpR(rhs, x_tmp);
                ResidualOp(x_tmp, y_tmp);
                ProjectOpL(y_tmp, z_tmp);
                auto t_end     = std::chrono::high_resolution_clock::now();
                mat.m_optimer += std::chrono::duration<double>(t_end - t_start).count();
                mat.m_opcounter++;

                if(alpha == Scalar{1})
                    dst.noalias() += z_tmp;
                else
                    dst.noalias() += alpha * z_tmp;
            }
        }
    };
}

namespace grit::internal::precondition {
    template<typename Scalar>
    typename JacobiDavidsonOperator<Scalar>::VectorType JacobiDavidsonSolver(JacobiDavidsonOperator<Scalar>                            &matRepl,
                                                                             const typename JacobiDavidsonOperator<Scalar>::VectorType &rhs,
                                                                             IterativeLinearSolverConfig<Scalar>                       &cfg) {
        using PreconditionerType = IterativeLinearSolverPreconditioner<JacobiDavidsonOperator<Scalar>>;
        using DefSolverType      = Eigen::ConjugateGradient_ST<JacobiDavidsonOperator<Scalar>, Eigen::Upper | Eigen::Lower, PreconditionerType>;
        using IndSolverType      = std::conditional_t<is_std_complex<Scalar>::value, Eigen::BiCGSTAB_ST<JacobiDavidsonOperator<Scalar>, PreconditionerType>,
                                                      Eigen::MINRES_ST<JacobiDavidsonOperator<Scalar>, Eigen::Upper | Eigen::Lower, PreconditionerType>>;
        std::variant<IndSolverType, DefSolverType> solverVariant;
        if(cfg.matdef == MatDef::IND)
            solverVariant.template emplace<0>();
        else
            solverVariant.template emplace<1>();

        typename JacobiDavidsonOperator<Scalar>::VectorType res;
        auto                                                run = [&](auto &solver) {
            auto t_start = std::chrono::high_resolution_clock::now();

            solver.setMaxIterations(cfg.maxiters);
            solver.setTolerance(cfg.tolerance);
            solver.preconditioner().attach(&matRepl, &cfg);
            solver.compute(matRepl);
            if(cfg.initialGuess.size() == rhs.size()) {
                res = solver.solveWithGuess(rhs, cfg.initialGuess);
            } else {
                res = solver.solve(rhs);
            }
            auto t_end = std::chrono::high_resolution_clock::now();

            cfg.result.iters        += solver.iterations();
            cfg.result.matvecs      += matRepl.iterations();
            cfg.result.precond      += solver.preconditioner().iterations();
            cfg.result.time         += std::chrono::duration<double>(t_end - t_start).count();
            cfg.result.time_matvecs += matRepl.elapsed_time();
            cfg.result.time_precond += solver.preconditioner().elapsed_time();

            cfg.result.error = solver.error();
            cfg.result.info  = solver.info();
            if(std::isnan(solver.error())) throw std::runtime_error("NaN in solver");
            return res;
        };

        std::visit(run, solverVariant);
        return res;
    }
}

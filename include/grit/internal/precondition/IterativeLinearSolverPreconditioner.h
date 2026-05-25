#pragma once

#include "IterativeLinearSolverConfig.h"
#include <cassert>
#include <chrono>
#include <Eigen/Core>
#include <stdexcept>

namespace grit::internal::precondition {
    template<typename MatrixLikeType>
    class IterativeLinearSolverPreconditioner {
        private:
        using Scalar     = typename MatrixLikeType::Scalar;
        using RealScalar = decltype(std::real(std::declval<Scalar>()));
        using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        protected:
        mutable Eigen::Index m_iterations    = 0;
        mutable double       m_time_elapsed  = 0.0;
        bool                 m_isInitialized = false;

        const MatrixLikeType                      *matrix = nullptr;
        const IterativeLinearSolverConfig<Scalar> *config = nullptr;

        mutable VectorType y;
        mutable VectorType z;
        mutable VectorType temp;

        public:
        using StorageIndex = typename VectorType::StorageIndex;
        enum { ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic };

        IterativeLinearSolverPreconditioner() = default;

        void attach(const MatrixLikeType *matrix_, const IterativeLinearSolverConfig<Scalar> *config_) {
            matrix = matrix_;
            config = config_;
            if(matrix != nullptr && config != nullptr) { m_isInitialized = true; }
        }

        template<typename MatType>
        explicit IterativeLinearSolverPreconditioner(const MatType &) {}
        Eigen::Index    iterations() const { return m_iterations; }
        double          elapsed_time() const { return m_time_elapsed; }
        double          time_jacobi() const { return 0.0; }
        double          time_chebyshev() const { return 0.0; }
        EIGEN_CONSTEXPR Eigen::Index rows() const noexcept { return matrix->rows(); }
        EIGEN_CONSTEXPR Eigen::Index cols() const noexcept { return matrix->cols(); }

        template<typename MatType>
        IterativeLinearSolverPreconditioner &analyzePattern(const MatType &) {
            return *this;
        }

        template<typename MatType>
        IterativeLinearSolverPreconditioner &factorize(const MatType &) {
            return *this;
        }

        template<typename MatType>
        IterativeLinearSolverPreconditioner &compute(const MatType &) {
            return *this;
        }

        template<typename Rhs, typename Dest>
        void _solve_impl(const Rhs &b, Dest &x) const {
            auto t_start = std::chrono::high_resolution_clock::now();

            if(!config->preconditioner_apply) {
                x = b;
                m_iterations++;
                auto t_end      = std::chrono::high_resolution_clock::now();
                m_time_elapsed += std::chrono::duration<double>(t_end - t_start).count();
                return;
            }

            y.resize(b.rows());
            z.resize(b.rows());
            temp.resize(b.rows());

            if constexpr(MatrixLikeType::has_projector_op) {
                matrix->ProjectOpL(b, y);
            } else {
                y = b;
            }

            config->preconditioner_apply(y, z, config->theta);

            if constexpr(MatrixLikeType::has_projector_op) {
                matrix->ProjectOpR(z, temp);
                x = temp;
            } else {
                x = z;
            }
            assert(x.allFinite());

            m_iterations++;
            auto t_end      = std::chrono::high_resolution_clock::now();
            m_time_elapsed += std::chrono::duration<double>(t_end - t_start).count();
        }

        template<typename Rhs>
        inline const Eigen::Solve<IterativeLinearSolverPreconditioner, Rhs> solve(const Eigen::MatrixBase<Rhs> &b) const {
            eigen_assert(m_isInitialized && "IterativeLinearSolverPreconditioner is not initialized.");
            eigen_assert(b.rows() == rows() && "Size mismatchs");
            eigen_assert(b.allFinite() && "All elements in b must be finite");
            return Eigen::Solve<IterativeLinearSolverPreconditioner, Rhs>(*this, b.derived());
        }

        Eigen::ComputationInfo info() const { return Eigen::Success; }
    };
}

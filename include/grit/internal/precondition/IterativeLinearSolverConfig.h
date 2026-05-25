#pragma once

#include <complex>
#include <Eigen/Core>
#include <functional>

namespace grit {
    enum class MatDef { IND, SEMI, DEF };

    template<typename Scalar>
    struct IterativeLinearSolverConfig {
        using Real                    = decltype(std::real(std::declval<Scalar>()));
        using VectorType              = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using MatrixType              = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using PreconditionerApplyFunc = std::function<void(const Eigen::Ref<const VectorType> &, Eigen::Ref<VectorType>, Real)>;

        struct Result {
            Eigen::Index           iters        = 0;
            Eigen::Index           matvecs      = 0;
            Eigen::Index           precond      = 0;
            double                 time         = 0;
            double                 time_matvecs = 0;
            double                 time_precond = 0;
            Real                   error        = 0;
            Eigen::ComputationInfo info         = Eigen::ComputationInfo::NoConvergence;

            void    add_latest(const Result &other) { *this += other; }
            Result &operator+=(const Result &other) {
                iters        += other.iters;
                matvecs      += other.matvecs;
                precond      += other.precond;
                time         += other.time;
                time_matvecs += other.time_matvecs;
                time_precond += other.time_precond;
                error         = other.error;
                info          = other.info;
                return *this;
            }
            void copy_latest(const Result &other) {
                iters        = other.iters;
                matvecs      = other.matvecs;
                precond      = other.precond;
                time         = other.time;
                time_matvecs = other.time_matvecs;
                time_precond = other.time_precond;
                error        = other.error;
                info         = other.info;
            }
            void reset() { *this = {}; }
        };

        long                    maxiters     = 1000;
        Real                    tolerance    = Real{0.1f};
        Real                    theta        = Real{0};
        MatDef                  matdef       = MatDef::IND;
        MatrixType              initialGuess = {};
        PreconditionerApplyFunc preconditioner_apply;
        mutable Result          result = {};
    };
}

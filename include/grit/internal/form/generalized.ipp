#pragma once

#include <grit/form/generalized.h>
#include <stdexcept>

namespace grit::form {
    template<typename Scalar>
    generalized<Scalar>::generalized(Eigen::Index nev, Eigen::Index ncv, OptRitz ritz, const MatrixType &V, MatVec<Scalar> &A, MatVec<Scalar> &B,
                                     spdlog::level::level_enum logLevel_)
        : base<Scalar>(nev, ncv, ritz, V, A, logLevel_), B(B) {}

    template<typename Scalar>
    std::string_view generalized<Scalar>::form_name() const {
        return "GENERALIZED";
    }

    template<typename Scalar>
    bool generalized<Scalar>::is_generalized_problem() const {
        return true;
    }

    template<typename Scalar>
    typename generalized<Scalar>::MatrixType generalized<Scalar>::MultB(const Eigen::Ref<const MatrixType> &X) {
        auto token_matvecs  = status.time_matvecs.tic_token();
        status.num_matvecs += X.cols();
        return B.MultBX(X);
    }

    template<typename Scalar>
    typename generalized<Scalar>::MatrixType generalized<Scalar>::MultB_inner(const Eigen::Ref<const MatrixType> &X) {
        return B.MultBX(X);
    }

    template<typename Scalar>
    void generalized<Scalar>::diagonalizeT() {
        T1 = Q.adjoint() * AQ;
        T2 = Q.adjoint() * BQ;
        T1 = (T1 + T1.adjoint()) * half;
        T2 = (T2 + T2.adjoint()) * half;
        Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType> es(T1, T2);
        if(es.info() != Eigen::Success) throw std::runtime_error("diagonalizeT: generalized eigensolver failed");
        T_evals = es.eigenvalues();
        T_evecs = es.eigenvectors();
    }

}

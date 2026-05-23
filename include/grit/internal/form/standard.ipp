#pragma once

#include <grit/form/standard.h>
#include <stdexcept>

namespace grit::form {
    template<typename Scalar>
    std::string_view standard<Scalar>::form_name() const {
        return "STANDARD";
    }

    template<typename Scalar>
    typename standard<Scalar>::MatrixType standard<Scalar>::MultB(const Eigen::Ref<const MatrixType> &X) {
        auto token_matvecs  = status.time_matvecs.tic_token();
        status.num_matvecs += X.cols();
        return X;
    }

    template<typename Scalar>
    void standard<Scalar>::diagonalizeT() {
        T1 = Q.adjoint() * AQ;
        T1 = (T1 + T1.adjoint()) * half;
        T2 = MatrixType::Identity(T1.rows(), T1.cols());
        Eigen::SelfAdjointEigenSolver<MatrixType> es(T1);
        if(es.info() != Eigen::Success) throw std::runtime_error("diagonalizeT: eigensolver failed");
        T_evals = es.eigenvalues();
        T_evecs = es.eigenvectors();
    }
}

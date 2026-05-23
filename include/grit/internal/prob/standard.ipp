#pragma once

#include <grit/prob/standard.h>
#include <stdexcept>

namespace grit::standard {
    template<typename Scalar_>
    problem<Scalar_>::problem(MatVec<Scalar> &A_) {
        set_A(A_);
    }

    template<typename Scalar_>
    void problem<Scalar_>::set_A(MatVec<Scalar> &A_) {
        A = &A_;
        if(size_) A->set_size(*size_);
        if(A_op_norm) A->set_op_norm(*A_op_norm);
    }

    template<typename Scalar_>
    bool problem<Scalar_>::has_A() const {
        return A != nullptr;
    }

    template<typename Scalar_>
    MatVec<typename problem<Scalar_>::Scalar> &problem<Scalar_>::get_A() const {
        if(A == nullptr) throw std::runtime_error("grit::standard::problem requires operator A");
        if(size_ && A->get_size() != *size_) throw std::runtime_error("grit::standard::problem size does not match A");
        if(A->get_size() <= 0) throw std::runtime_error("grit::standard::problem has invalid size");
        return *A;
    }

    template<typename Scalar_>
    void problem<Scalar_>::set_size(Eigen::Index size) {
        size_ = size;
        if(A) A->set_size(size);
    }

    template<typename Scalar_>
    Eigen::Index problem<Scalar_>::get_size() const {
        if(size_) return *size_;
        if(A) return A->get_size();
        return 0;
    }

    template<typename Scalar_>
    void problem<Scalar_>::set_op_norm(RealScalar op_norm) {
        A_op_norm = op_norm;
        if(A) A->set_op_norm(op_norm);
    }

    template<typename Scalar_>
    std::optional<typename problem<Scalar_>::RealScalar> problem<Scalar_>::get_op_norm() const {
        return A_op_norm;
    }

}

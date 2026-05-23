#pragma once

#include <grit/prob/generalized.h>
#include <stdexcept>

namespace grit::generalized {
    template<typename Scalar_>
    problem<Scalar_>::problem(MatVec<Scalar> &A_, MatVec<Scalar> &B_) {
        set_A(A_);
        set_B(B_);
    }

    template<typename Scalar_>
    void problem<Scalar_>::set_A(MatVec<Scalar> &A_) {
        A = &A_;
        if(size_) A->set_size(*size_);
        if(A_op_norm) A->set_op_norm(*A_op_norm);
    }

    template<typename Scalar_>
    void problem<Scalar_>::set_B(MatVec<Scalar> &B_) {
        B = &B_;
        if(size_) B->set_size(*size_);
        if(B_op_norm) B->set_op_norm(*B_op_norm);
    }

    template<typename Scalar_>
    bool problem<Scalar_>::has_A() const {
        return A != nullptr;
    }

    template<typename Scalar_>
    bool problem<Scalar_>::has_B() const {
        return B != nullptr;
    }

    template<typename Scalar_>
    MatVec<typename problem<Scalar_>::Scalar> &problem<Scalar_>::get_A() const {
        if(A == nullptr) throw std::runtime_error("grit::generalized::problem requires operator A");
        if(A->get_size() <= 0) throw std::runtime_error("grit::generalized::problem has invalid A size");
        if(size_ && A->get_size() != *size_) throw std::runtime_error("grit::generalized::problem size does not match A");
        if(B && B->get_size() > 0 && A->get_size() != B->get_size()) throw std::runtime_error("grit::generalized::problem A and B sizes differ");
        return *A;
    }

    template<typename Scalar_>
    MatVec<typename problem<Scalar_>::Scalar> &problem<Scalar_>::get_B() const {
        if(B == nullptr) throw std::runtime_error("grit::generalized::problem requires operator B");
        if(B->get_size() <= 0) throw std::runtime_error("grit::generalized::problem has invalid B size");
        if(size_ && B->get_size() != *size_) throw std::runtime_error("grit::generalized::problem size does not match B");
        if(A && A->get_size() > 0 && A->get_size() != B->get_size()) throw std::runtime_error("grit::generalized::problem A and B sizes differ");
        return *B;
    }

    template<typename Scalar_>
    void problem<Scalar_>::set_size(Eigen::Index size) {
        size_ = size;
        if(A) A->set_size(size);
        if(B) B->set_size(size);
    }

    template<typename Scalar_>
    Eigen::Index problem<Scalar_>::get_size() const {
        if(size_) return *size_;
        if(A) return A->get_size();
        if(B) return B->get_size();
        return 0;
    }

    template<typename Scalar_>
    void problem<Scalar_>::set_A_op_norm(RealScalar op_norm) {
        A_op_norm = op_norm;
        if(A) A->set_op_norm(op_norm);
    }

    template<typename Scalar_>
    void problem<Scalar_>::set_B_op_norm(RealScalar op_norm) {
        B_op_norm = op_norm;
        if(B) B->set_op_norm(op_norm);
    }

    template<typename Scalar_>
    std::optional<typename problem<Scalar_>::RealScalar> problem<Scalar_>::get_A_op_norm() const {
        return A_op_norm;
    }

    template<typename Scalar_>
    std::optional<typename problem<Scalar_>::RealScalar> problem<Scalar_>::get_B_op_norm() const {
        return B_op_norm;
    }

}

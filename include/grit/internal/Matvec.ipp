#pragma once

#include <grit/Matvec.h>
#include <stdexcept>

namespace grit {
    template<typename Scalar_>
    Matvec<Scalar_>::Matvec(Eigen::Index size_, MultFunc mult_) : size(size_), mult_callback(std::move(mult_)) {}

    template<typename Scalar_>
    Matvec<Scalar_>::Matvec(Matvec &&other) noexcept = default;

    template<typename Scalar_>
    Matvec<Scalar_> &Matvec<Scalar_>::operator=(Matvec &&other) noexcept = default;

    template<typename Scalar_>
    Matvec<Scalar_>::~Matvec() = default;

    template<typename Scalar_>
    void Matvec<Scalar_>::set_mult(MultFunc mult_) {
        mult_callback = std::move(mult_);
    }

    template<typename Scalar_>
    void Matvec<Scalar_>::set_preconditioner_update(PreconditionerUpdateFunc preconditioner_update_) {
        preconditioner_update_callback = std::move(preconditioner_update_);
    }

    template<typename Scalar_>
    void Matvec<Scalar_>::set_preconditioner_apply(PreconditionerApplyFunc preconditioner_apply_) {
        preconditioner_apply_callback = std::move(preconditioner_apply_);
    }

    template<typename Scalar_>
    void Matvec<Scalar_>::set_op_norm(RealScalar op_norm_) {
        op_norm = op_norm_;
    }

    template<typename Scalar_>
    void Matvec<Scalar_>::set_size(Eigen::Index size_) {
        size = size_;
    }

    template<typename Scalar_>
    int Matvec<Scalar_>::rows() const {
        return static_cast<int>(size);
    }

    template<typename Scalar_>
    int Matvec<Scalar_>::cols() const {
        return static_cast<int>(size);
    }

    template<typename Scalar_>
    Eigen::Index Matvec<Scalar_>::get_size() const {
        return size;
    }

    template<typename Scalar_>
    typename Matvec<Scalar_>::RealScalar Matvec<Scalar_>::get_op_norm() const {
        return op_norm.value_or(RealScalar{1});
    }

    template<typename Scalar_>
    bool Matvec<Scalar_>::has_mult() const {
        return static_cast<bool>(mult_callback);
    }

    template<typename Scalar_>
    bool Matvec<Scalar_>::has_preconditioner_apply() const {
        return static_cast<bool>(preconditioner_apply_callback);
    }

    template<typename Scalar_>
    typename Matvec<Scalar_>::MatrixType Matvec<Scalar_>::mult(const Eigen::Ref<const MatrixType> &X) const {
        if(!mult_callback) throw std::runtime_error("Matvec::mult callback is not set");
        auto token_apply = t_mult->tic_token();
        num_mv += X.cols();
        return mult_callback(X);
    }

    template<typename Scalar_>
    void Matvec<Scalar_>::preconditioner_update(RealScalar theta) const {
        if(preconditioner_update_callback) preconditioner_update_callback(theta);
    }

    template<typename Scalar_>
    void Matvec<Scalar_>::preconditioner_apply(const Eigen::Ref<const VectorType> &x, Eigen::Ref<VectorType> y, RealScalar theta) const {
        if(!preconditioner_apply_callback) {
            y = x;
            return;
        }
        auto token_precond = t_precond->tic_token();
        preconditioner_apply_callback(x, y, theta);
        num_pc++;
    }

    template<typename Scalar_>
    void Matvec<Scalar_>::reset() {
        num_mv = 0;
        num_pc = 0;
    }

    template<typename Scalar>
    Matvec<Scalar> matvec(Eigen::Index size, typename Matvec<Scalar>::MultFunc callback) {
        return Matvec<Scalar>(size, std::move(callback));
    }

    template<typename Scalar>
    Matvec<Scalar> matvec(Eigen::Index size, ptr_t, typename Matvec<Scalar>::PtrMultFunc callback) {
        using MatrixType = typename Matvec<Scalar>::MatrixType;
        auto wrapped     = [size, callback = std::move(callback)](const Eigen::Ref<const MatrixType> &X) mutable -> MatrixType {
            MatrixType Y(size, X.cols());
            callback(X.data(), Y.data(), X.rows(), X.cols());
            return Y;
        };
        return Matvec<Scalar>(size, std::move(wrapped));
    }
}

#pragma once

#include <grit/MatVec.h>
#include <stdexcept>

namespace grit {
    template<typename Scalar_>
    MatVec<Scalar_>::MatVec(Eigen::Index size_, MultAXFunc MultAX_) : size(size_), MultAX_callback(std::move(MultAX_)) {}

    template<typename Scalar_>
    MatVec<Scalar_>::MatVec(MatVec &&other) noexcept = default;

    template<typename Scalar_>
    MatVec<Scalar_> &MatVec<Scalar_>::operator=(MatVec &&other) noexcept = default;

    template<typename Scalar_>
    MatVec<Scalar_>::~MatVec() = default;

    template<typename Scalar_>
    void MatVec<Scalar_>::set_MultAX(MultAXFunc MultAX_) {
        MultAX_callback = std::move(MultAX_);
    }

    template<typename Scalar_>
    void MatVec<Scalar_>::set_MultBX(MultAXFunc MultBX_) {
        MultBX_callback = std::move(MultBX_);
    }

    template<typename Scalar_>
    void MatVec<Scalar_>::set_preconditioner_update(PreconditionerUpdateFunc preconditioner_update_) {
        preconditioner_update_callback = std::move(preconditioner_update_);
    }

    template<typename Scalar_>
    void MatVec<Scalar_>::set_preconditioner_apply(PreconditionerApplyFunc preconditioner_apply_) {
        preconditioner_apply_callback = std::move(preconditioner_apply_);
    }

    template<typename Scalar_>
    void MatVec<Scalar_>::set_op_norm(RealScalar op_norm_) {
        op_norm = op_norm_;
    }

    template<typename Scalar_>
    void MatVec<Scalar_>::set_size(Eigen::Index size_) {
        size = size_;
    }

    template<typename Scalar_>
    int MatVec<Scalar_>::rows() const {
        return static_cast<int>(size);
    }

    template<typename Scalar_>
    int MatVec<Scalar_>::cols() const {
        return static_cast<int>(size);
    }

    template<typename Scalar_>
    Eigen::Index MatVec<Scalar_>::get_size() const {
        return size;
    }

    template<typename Scalar_>
    typename MatVec<Scalar_>::RealScalar MatVec<Scalar_>::get_op_norm() const {
        return op_norm.value_or(RealScalar{1});
    }

    template<typename Scalar_>
    bool MatVec<Scalar_>::has_preconditioner_apply() const {
        return static_cast<bool>(preconditioner_apply_callback);
    }

    template<typename Scalar_>
    typename MatVec<Scalar_>::MatrixType MatVec<Scalar_>::MultAX(const Eigen::Ref<const MatrixType> &X) const {
        if(!MultAX_callback) throw std::runtime_error("MatVec::MultAX callback is not set");
        num_mv += X.cols();
        return MultAX_callback(X);
    }

    template<typename Scalar_>
    typename MatVec<Scalar_>::MatrixType MatVec<Scalar_>::MultBX(const Eigen::Ref<const MatrixType> &X) const {
        if(MultBX_callback) {
            num_mv += X.cols();
            return MultBX_callback(X);
        }
        return X;
    }

    template<typename Scalar_>
    void MatVec<Scalar_>::preconditioner_update(RealScalar theta) const {
        if(preconditioner_update_callback) preconditioner_update_callback(theta);
    }

    template<typename Scalar_>
    void MatVec<Scalar_>::preconditioner_apply(const Eigen::Ref<const VectorType> &x, Eigen::Ref<VectorType> y, RealScalar theta) const {
        if(!preconditioner_apply_callback) {
            y = x;
            return;
        }
        auto token_precond = t_multPc->tic_token();
        preconditioner_apply_callback(x, y, theta);
        num_pc++;
    }

    template<typename Scalar_>
    void MatVec<Scalar_>::reset() {
        num_mv = 0;
        num_op = 0;
        num_pc = 0;
    }

    template<typename Scalar>
    MatVec<Scalar> matvec(Eigen::Index size, typename MatVec<Scalar>::MultAXFunc callback) {
        MatVec<Scalar> mv(size, callback);
        mv.set_MultBX(std::move(callback));
        return mv;
    }

    template<typename Scalar>
    MatVec<Scalar> matvec(Eigen::Index size, raw_t, typename MatVec<Scalar>::RawMultAXFunc callback) {
        using MatrixType = typename MatVec<Scalar>::MatrixType;
        auto wrapped     = [size, callback = std::move(callback)](const Eigen::Ref<const MatrixType> &X) mutable -> MatrixType {
            MatrixType Y(size, X.cols());
            callback(X.data(), Y.data(), X.rows(), X.cols());
            return Y;
        };
        MatVec<Scalar> mv(size, wrapped);
        mv.set_MultBX(std::move(wrapped));
        return mv;
    }
}

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
    void MatVec<Scalar_>::set_MultPX(MultPXFunc MultPX_) {
        MultPX_callback = std::move(MultPX_);
    }

    template<typename Scalar_>
    void MatVec<Scalar_>::set_CalcPc(CalcPcFunc CalcPc_) {
        CalcPc_callback = std::move(CalcPc_);
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
    typename MatVec<Scalar_>::MatrixType MatVec<Scalar_>::MultPX(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const VectorReal> &evals) {
        if(MultPX_callback) {
            num_pc += X.cols();
            return MultPX_callback(X, evals, std::nullopt);
        }
        return X;
    }

    template<typename Scalar_>
    void MatVec<Scalar_>::CalcPc(RealScalar shift) {
        if(CalcPc_callback) CalcPc_callback(shift);
    }

    template<typename Scalar_>
    void MatVec<Scalar_>::reset() {
        num_mv = 0;
        num_pc = 0;
        iLinSolvCfg.result.reset();
    }

    template<typename Scalar_>
    void MatVec<Scalar_>::set_iterativeLinearSolverConfig(const IterativeLinearSolverConfig<Scalar> &cfg) {
        iLinSolvCfg = cfg;
    }

    template<typename Scalar_>
    void MatVec<Scalar_>::set_iterativeLinearSolverConfig(long maxiters, RealScalar tolerance, MatDef matdef) {
        iLinSolvCfg.maxiters  = maxiters;
        iLinSolvCfg.tolerance = tolerance;
        iLinSolvCfg.matdef    = matdef;
    }

    template<typename Scalar_>
    IterativeLinearSolverConfig<typename MatVec<Scalar_>::Scalar> &MatVec<Scalar_>::get_iterativeLinearSolverConfig() {
        return iLinSolvCfg;
    }

    template<typename Scalar_>
    const IterativeLinearSolverConfig<typename MatVec<Scalar_>::Scalar> &MatVec<Scalar_>::get_iterativeLinearSolverConfig() const {
        return iLinSolvCfg;
    }

    template<typename Scalar_>
    void MatVec<Scalar_>::set_jcbMaxBlockSize(std::optional<long> jcbSize) {
        jcbMaxBlockSize = jcbSize.value_or(-1);
    }

    template<typename Scalar_>
    void MatVec<Scalar_>::set_jcbMaxBlockSize(long jcbSize) {
        jcbMaxBlockSize = jcbSize;
    }

    template<typename Scalar_>
    void MatVec<Scalar_>::set_jcbOverlapSize(std::optional<long> size_) {
        jcbOverlapSize = size_.value_or(0);
    }

    template<typename Scalar_>
    void MatVec<Scalar_>::set_jcbOverlapSize(long size_) {
        jcbOverlapSize = size_;
    }

    template<typename Scalar_>
    void MatVec<Scalar_>::set_jcbNumPasses(std::optional<long> size_) {
        jcbNumPasses = size_.value_or(1);
    }

    template<typename Scalar_>
    void MatVec<Scalar_>::set_jcbNumPasses(long size_) {
        jcbNumPasses = size_;
    }

    template<typename Scalar_>
    long MatVec<Scalar_>::get_jcbMaxBlockSize() const {
        return jcbMaxBlockSize;
    }

    template<typename Scalar_>
    long MatVec<Scalar_>::get_jcbOverlapSize() const {
        return jcbOverlapSize;
    }

    template<typename Scalar_>
    long MatVec<Scalar_>::get_jcbNumPasses() const {
        return jcbNumPasses;
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
        auto wrapped = [size, callback = std::move(callback)](const Eigen::Ref<const MatrixType> &X) mutable -> MatrixType {
            MatrixType Y(size, X.cols());
            callback(X.data(), Y.data(), X.rows(), X.cols());
            return Y;
        };
        MatVec<Scalar> mv(size, wrapped);
        mv.set_MultBX(std::move(wrapped));
        return mv;
    }
}

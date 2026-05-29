#pragma once

#include <algorithm>
#include <stdexcept>
#include "grit/algo/lanczos.h"

namespace grit::algo {
    template<typename Scalar, grit::Form form_>
    const typename lanczos<Scalar, form_>::MatrixType &lanczos<Scalar, form_>::default_initial_guess() {
        static const MatrixType guess;
        return guess;
    }

    template<typename Scalar, grit::Form form_>
    lanczos<Scalar, form_>::lanczos(Matvec<Scalar> &A) requires(form_ == grit::Form::STANDARD)
        : Base(default_initial_guess(), A) {
        this->bind_config(config);
        config.nev              = 1;
        config.block_size       = 1;
        config.ncv              = std::min<Eigen::Index>(8, std::max<Eigen::Index>(1, this->N));
        config.max_basis_blocks = config.ncv;
    }

    template<typename Scalar, grit::Form form_>
    lanczos<Scalar, form_>::lanczos(Matvec<Scalar> &A, Matvec<Scalar> &B) requires(form_ == grit::Form::GENERALIZED)
        : Base(default_initial_guess(), A, B) {
        this->bind_config(config);
        config.nev              = 1;
        config.block_size       = 1;
        config.ncv              = std::min<Eigen::Index>(8, std::max<Eigen::Index>(1, this->N));
        config.max_basis_blocks = config.ncv;
    }

    template<typename Scalar, grit::Form form_>
    void lanczos<Scalar, form_>::set_initial_guess(const MatrixType &guess) {
        initial_guess_ptr = &guess;
    }

    template<typename Scalar, grit::Form form_>
    void lanczos<Scalar, form_>::clear_initial_guess() {
        initial_guess_ptr = nullptr;
    }

    template<typename Scalar, grit::Form form_>
    bool lanczos<Scalar, form_>::has_initial_guess() const {
        return initial_guess_ptr != nullptr;
    }

    template<typename Scalar, grit::Form form_>
    const typename lanczos<Scalar, form_>::MatrixType &lanczos<Scalar, form_>::initial_guess() const {
        if(initial_guess_ptr) return *initial_guess_ptr;
        return empty_initial_guess;
    }

    template<typename Scalar, grit::Form form_>
    void lanczos<Scalar, form_>::assert_operator_config() const requires(form_ == grit::Form::STANDARD)
    {
        if(this->A.get_size() <= 0) throw std::runtime_error("lanczos requires operator A with positive size");
        if(config.use_b_inner_product) throw std::runtime_error("use_b_inner_product requires a generalized problem");
    }

    template<typename Scalar, grit::Form form_>
    void lanczos<Scalar, form_>::assert_operator_config() const requires(form_ == grit::Form::GENERALIZED)
    {
        if(this->A.get_size() <= 0) throw std::runtime_error("lanczos requires operator A with positive size");
        auto &B = this->B->get();
        if(B.get_size() <= 0) throw std::runtime_error("lanczos requires operator B with positive size");
        if(this->A.get_size() != B.get_size()) throw std::runtime_error("lanczos requires operators A and B to have matching sizes");
    }

    template<typename Scalar, grit::Form form_>
    void lanczos<Scalar, form_>::assert_config() const {
        assert_operator_config();

        if(config.nev < 1) throw std::runtime_error("lanczos config error: nev must be at least 1");
        if(config.block_size < 1) throw std::runtime_error("lanczos config error: block_size must be at least 1");
        if(config.ncv < config.nev) throw std::runtime_error("lanczos config error: ncv must be at least nev");
        if(config.ncv > this->N) throw std::runtime_error("lanczos config error: ncv must not exceed the operator size");
        if(config.block_size > config.ncv) throw std::runtime_error("lanczos config error: block_size must not exceed ncv");
        if(config.ncv % config.block_size != 0) throw std::runtime_error("lanczos config error: ncv must be divisible by block_size");
        if(config.max_basis_blocks < 1) throw std::runtime_error("lanczos config error: max_basis_blocks must be at least 1");
        if(config.max_basis_blocks * config.block_size != config.ncv) {
            throw std::runtime_error("lanczos config error: max_basis_blocks * block_size must equal ncv");
        }
        if(config.max_iters == 0) throw std::runtime_error("lanczos config error: max_iters must be positive or negative for unlimited");
        if(config.max_matvecs == 0) throw std::runtime_error("lanczos config error: max_matvecs must be positive or negative for unlimited");
        if(config.tol <= RealScalar{0}) throw std::runtime_error("lanczos config error: tol must be positive");
        if(config.tol_rnorm_relative < RealScalar{0}) throw std::runtime_error("lanczos config error: tol_rnorm_relative must be nonnegative");
        if(config.sat_eigval_threshold < RealScalar{0}) throw std::runtime_error("lanczos config error: sat_eigval_threshold must be nonnegative");
        if(config.sat_rnorm_threshold < RealScalar{0}) throw std::runtime_error("lanczos config error: sat_rnorm_threshold must be nonnegative");
        if(has_initial_guess()) {
            if(initial_guess().rows() != this->N) throw std::runtime_error("lanczos config error: initial guess row count must match the operator size");
            if(initial_guess().cols() < 1) throw std::runtime_error("lanczos config error: initial guess must have at least one column");
        }
    }

    template<typename Scalar, grit::Form form_>
    void lanczos<Scalar, form_>::run() {
        assert_config();

        this->setLogger(config.log_level, std::string("grit|") + std::string(this->form_name()));
        this->V = initial_guess();
        beta_stalled = false;
        clear_result();
        Base::run();
    }

    template<typename Scalar, grit::Form form_>
    void lanczos<Scalar, form_>::write_Q_next_B_DGKS(Eigen::Index i) {
        for(int rep = 0; rep < 2; ++rep) {
            MatrixType QjW;
            for(Eigen::Index j = i; j >= 0; --j) {
                auto Qj = Q.middleCols(j * this->cfg().block_size, this->cfg().block_size);
                QjW     = Qj.adjoint() * W;
                W.noalias() -= Qj * QjW;
            }
        }

        hhqr.compute(W);
        Q.middleCols((i + 1) * this->cfg().block_size, this->cfg().block_size) =
            hhqr.householderQ().setLength(W.cols()) * MatrixType::Identity(N, this->cfg().block_size);
        B_block = hhqr.matrixQR().topLeftCorner(this->cfg().block_size, this->cfg().block_size).template triangularView<Eigen::Upper>();
    }

    template<typename Scalar, grit::Form form_>
    void lanczos<Scalar, form_>::build() {
        const Eigen::Index b = this->cfg().block_size;
        const Eigen::Index m = config.max_basis_blocks;

        beta_stalled = false;
        if(V.cols() != b) throw std::runtime_error("lanczos build error: V must have block_size columns");

        Q  = V;
        AQ = AV;
        if constexpr(form_ == grit::Form::GENERALIZED) BQ = BV;
        else BQ = Q;

        MatrixType Q_prev;
        A_block.resize(b, b);
        B_block.resize(b, b);
        T.setZero(std::min(N, m * b), std::min(N, m * b));

        for(Eigen::Index i = 0; i + 1 < m && Q.cols() + b <= N; ++i) {
            auto Q_cur = Q.rightCols(b);
            W          = MultA(Q_cur);

            A_block = Q_cur.adjoint() * W;
            W.noalias() -= Q_cur * A_block;
            if(Q_prev.cols() == b) {
                B_block = Q_prev.adjoint() * W;
                W.noalias() -= Q_prev * B_block.adjoint();
            } else {
                B_block.setZero();
            }

            auto min_rnorm    = W.colwise().norm().minCoeff();
            auto breakdownTol = eps * 10 * std::max(A_block.norm(), B_block.norm());
            if(min_rnorm < breakdownTol) {
                beta_stalled = true;
                break;
            }

            if(A.has_preconditioner_apply() && T_evals.size() >= b) {
                auto select_b = this->get_ritz_indices(this->cfg().ritz, 0, b, T_evals);
                VectorReal evals(b);
                for(Eigen::Index j = 0; j < b; ++j) evals(j) = T_evals(select_b[static_cast<size_t>(j)]);
                W = MultP(W, evals);
            }

            MatrixType AW = MultA(W);
            MatrixType BW = MatrixType();
            OrthMeta   meta;
            meta.maskPolicy = Base::MaskPolicy::COMPRESS;

            if constexpr(form_ == grit::Form::GENERALIZED) {
                BW = MultB(W);
                if(config.use_b_inner_product) {
                    block_bm_orthogonalize(Q, AQ, BQ, W, AW, BW, meta);
                    block_bm_orthonormalize(W, AW, BW, meta);
                } else {
                    block_l2_orthogonalize(Q, AQ, BQ, W, AW, BW, meta);
                    block_l2_orthonormalize(W, AW, BW, meta);
                }
            } else {
                block_l2_orthogonalize(Q, AQ, W, AW, meta);
                block_l2_orthonormalize(W, AW, meta);
                BW = W;
            }

            if(W.cols() < b) {
                beta_stalled = true;
                break;
            }

            Q_prev = Q_cur;
            Eigen::Index old_cols = Q.cols();
            Q.conservativeResize(Eigen::NoChange, old_cols + b);
            AQ.conservativeResize(Eigen::NoChange, old_cols + b);
            BQ.conservativeResize(Eigen::NoChange, old_cols + b);
            Q.rightCols(b)  = W.leftCols(b);
            AQ.rightCols(b) = AW.leftCols(b);
            BQ.rightCols(b) = BW.leftCols(b);
        }

        qBlocks = Q.cols() / b;
    }

    template<typename Scalar, grit::Form form_>
    void lanczos<Scalar, form_>::extractRitzVectors() {
        if(status.stopReason != StopReason::none) return;
        if(T_evals.size() < this->cfg().block_size || T_evecs.cols() == 0) {
            status.stopReason |= StopReason::subspace_exhausted;
            status.stopMessage.emplace_back("lanczos extractRitzVectors: projected problem is empty");
            return;
        }

        Eigen::Index k     = std::min<Eigen::Index>(this->cfg().block_size, T_evals.size());
        Eigen::Index nritz = std::min<Eigen::Index>(T_evals.size(), std::max({this->cfg().nev, this->cfg().block_size, k}));
        status.optIdx      = this->get_ritz_indices(this->cfg().ritz, 0, nritz, T_evals);
        if(status.optIdx.empty()) {
            status.stopReason |= StopReason::subspace_exhausted;
            status.stopMessage.emplace_back("lanczos extractRitzVectors: no Ritz indices available");
            return;
        }

        if constexpr(form_ == grit::Form::GENERALIZED) {
            Base::extractRitzVectors(status.optIdx, V, AV, BV, S, status.rNorms);
        } else {
            Base::extractRitzVectors(status.optIdx, V, AV, S, status.rNorms);
            BV = V;
        }

        K_prev = K;
        K      = V.leftCols(k);
        if(status.rNorms_init.size() != status.rNorms.size()) status.rNorms_init = status.rNorms;
    }

    template<typename Scalar, grit::Form form_>
    void lanczos<Scalar, form_>::updateStatus() {
        if(T_evals.size() < this->cfg().block_size) return;
        Base::updateStatus();
        if(beta_stalled && status.stopReason == StopReason::none) {
            status.stopReason |= StopReason::lanczos_beta_stalled;
            status.stopMessage.emplace_back("lanczos beta stalled: exhausted subspace search");
        }
    }

    template<typename Scalar, grit::Form form_>
    void lanczos<Scalar, form_>::run_user_callback() {
        if(config.user_callback) config.user_callback(*this);
    }
}

#pragma once

#include <algorithm>
#include <stdexcept>
#include "grit/algo/lobpcg.h"

namespace grit::algo {
    template<typename Scalar, grit::Form form_>
    const typename lobpcg<Scalar, form_>::MatrixType &lobpcg<Scalar, form_>::default_initial_guess() {
        static const MatrixType guess;
        return guess;
    }

    template<typename Scalar, grit::Form form_>
    lobpcg<Scalar, form_>::lobpcg(Matvec<Scalar> &A) requires(form_ == grit::Form::STANDARD)
        : Base(default_initial_guess(), A) {
        this->bind_config(config);
        config.nev                    = 1;
        config.block_size             = 1;
        config.ncv                    = std::min<Eigen::Index>(8, std::max<Eigen::Index>(1, this->N));
        config.max_basis_blocks       = config.ncv;
        config.max_extra_ritz_history = 1;
        config.max_ritz_residual_history = 1;
    }

    template<typename Scalar, grit::Form form_>
    lobpcg<Scalar, form_>::lobpcg(Matvec<Scalar> &A, Matvec<Scalar> &B) requires(form_ == grit::Form::GENERALIZED)
        : Base(default_initial_guess(), A, B) {
        this->bind_config(config);
        config.nev                    = 1;
        config.block_size             = 1;
        config.ncv                    = std::min<Eigen::Index>(8, std::max<Eigen::Index>(1, this->N));
        config.max_basis_blocks       = config.ncv;
        config.max_extra_ritz_history = 1;
        config.max_ritz_residual_history = 1;
    }

    template<typename Scalar, grit::Form form_>
    void lobpcg<Scalar, form_>::set_initial_guess(const MatrixType &guess) {
        initial_guess_ptr = &guess;
    }

    template<typename Scalar, grit::Form form_>
    void lobpcg<Scalar, form_>::clear_initial_guess() {
        initial_guess_ptr = nullptr;
    }

    template<typename Scalar, grit::Form form_>
    bool lobpcg<Scalar, form_>::has_initial_guess() const {
        return initial_guess_ptr != nullptr;
    }

    template<typename Scalar, grit::Form form_>
    const typename lobpcg<Scalar, form_>::MatrixType &lobpcg<Scalar, form_>::initial_guess() const {
        if(initial_guess_ptr) return *initial_guess_ptr;
        return empty_initial_guess;
    }

    template<typename Scalar, grit::Form form_>
    void lobpcg<Scalar, form_>::assert_operator_config() const requires(form_ == grit::Form::STANDARD)
    {
        if(this->A.get_size() <= 0) throw std::runtime_error("lobpcg requires operator A with positive size");
        if(config.use_b_inner_product) throw std::runtime_error("use_b_inner_product requires a generalized problem");
    }

    template<typename Scalar, grit::Form form_>
    void lobpcg<Scalar, form_>::assert_operator_config() const requires(form_ == grit::Form::GENERALIZED)
    {
        if(this->A.get_size() <= 0) throw std::runtime_error("lobpcg requires operator A with positive size");
        auto &B = this->B->get();
        if(B.get_size() <= 0) throw std::runtime_error("lobpcg requires operator B with positive size");
        if(this->A.get_size() != B.get_size()) throw std::runtime_error("lobpcg requires operators A and B to have matching sizes");
    }

    template<typename Scalar, grit::Form form_>
    void lobpcg<Scalar, form_>::assert_config() const {
        assert_operator_config();

        if(config.nev < 1) throw std::runtime_error("lobpcg config error: nev must be at least 1");
        if(config.block_size < 1) throw std::runtime_error("lobpcg config error: block_size must be at least 1");
        if(config.ncv < config.nev) throw std::runtime_error("lobpcg config error: ncv must be at least nev");
        if(config.ncv > this->N) throw std::runtime_error("lobpcg config error: ncv must not exceed the operator size");
        if(config.block_size > config.ncv) throw std::runtime_error("lobpcg config error: block_size must not exceed ncv");
        if(config.ncv % config.block_size != 0) throw std::runtime_error("lobpcg config error: ncv must be divisible by block_size");
        if(config.max_basis_blocks < 1) throw std::runtime_error("lobpcg config error: max_basis_blocks must be at least 1");
        if(config.max_basis_blocks * config.block_size != config.ncv) {
            throw std::runtime_error("lobpcg config error: max_basis_blocks * block_size must equal ncv");
        }
        if(config.max_extra_ritz_history < 0) throw std::runtime_error("lobpcg config error: max_extra_ritz_history must be nonnegative");
        if(config.max_ritz_residual_history < 0) throw std::runtime_error("lobpcg config error: max_ritz_residual_history must be nonnegative");
        if(config.max_iters == 0) throw std::runtime_error("lobpcg config error: max_iters must be positive or negative for unlimited");
        if(config.max_matvecs == 0) throw std::runtime_error("lobpcg config error: max_matvecs must be positive or negative for unlimited");
        if(config.tol <= RealScalar{0}) throw std::runtime_error("lobpcg config error: tol must be positive");
        if(config.tol_rnorm_relative < RealScalar{0}) throw std::runtime_error("lobpcg config error: tol_rnorm_relative must be nonnegative");
        if(config.sat_eigval_threshold < RealScalar{0}) throw std::runtime_error("lobpcg config error: sat_eigval_threshold must be nonnegative");
        if(config.sat_rnorm_threshold < RealScalar{0}) throw std::runtime_error("lobpcg config error: sat_rnorm_threshold must be nonnegative");
        if(has_initial_guess()) {
            if(initial_guess().rows() != this->N) throw std::runtime_error("lobpcg config error: initial guess row count must match the operator size");
            if(initial_guess().cols() < 1) throw std::runtime_error("lobpcg config error: initial guess must have at least one column");
        }
    }

    template<typename Scalar, grit::Form form_>
    void lobpcg<Scalar, form_>::run() {
        assert_config();

        this->setLogger(config.log_level, std::string("grit|") + std::string(this->form_name()));
        this->V = initial_guess();
        max_mBlocks = config.max_extra_ritz_history;
        max_sBlocks = config.max_ritz_residual_history;
        clear_result();
        Base::run();
    }

    template<typename Scalar, grit::Form form_>
    void lobpcg<Scalar, form_>::shift_blocks_right(Eigen::Ref<MatrixType> matrix, Eigen::Index offset_old, Eigen::Index offset_new, Eigen::Index extent) {
        if(extent <= 0 || offset_old == offset_new) return;
        auto from = matrix.middleCols(offset_old * this->cfg().block_size, extent * this->cfg().block_size);
        auto to   = matrix.middleCols(offset_new * this->cfg().block_size, extent * this->cfg().block_size);
        to        = from.eval();
    }

    template<typename Scalar, grit::Form form_>
    void lobpcg<Scalar, form_>::roll_blocks_left(Eigen::Ref<MatrixType> matrix, Eigen::Index offset, Eigen::Index extent) {
        for(Eigen::Index k = extent - 1; k > 0; --k) {
            auto K0 = matrix.middleCols((offset + k + 0) * this->cfg().block_size, this->cfg().block_size);
            auto K1 = matrix.middleCols((offset + k - 1) * this->cfg().block_size, this->cfg().block_size);
            K0      = K1;
        }
    }

    template<typename Scalar, grit::Form form_>
    std::pair<typename lobpcg<Scalar, form_>::VectorIdxT, typename lobpcg<Scalar, form_>::VectorIdxT> lobpcg<Scalar, form_>::selective_orthonormalize() {
        using Index = Eigen::Index;
        Index n_blocks = Q.cols() / this->cfg().block_size;
        MatrixType Gram = Q.adjoint() * Q;

        std::vector<Index> needs_reortho;
        for(Index blk = 0; blk < n_blocks; ++blk) {
            Index col_start = blk * this->cfg().block_size;
            bool  bad       = false;
            for(Index prev_blk = 0; prev_blk < blk; ++prev_blk) {
                Index prev_col_start = prev_blk * this->cfg().block_size;
                auto  G_block        = Gram.block(prev_col_start, col_start, this->cfg().block_size, this->cfg().block_size);
                if(G_block.cwiseAbs().maxCoeff() > this->orthTol) {
                    bad = true;
                    break;
                }
            }
            if(!bad) {
                auto       G_diag = Gram.block(col_start, col_start, this->cfg().block_size, this->cfg().block_size);
                MatrixType I      = MatrixType::Identity(this->cfg().block_size, this->cfg().block_size);
                if((G_diag - I).cwiseAbs().maxCoeff() > this->normTol) bad = true;
            }
            if(bad) needs_reortho.push_back(blk);
        }

        VectorIdxT active_block_mask = VectorIdxT::Ones(n_blocks);
        VectorIdxT change_block_mask = VectorIdxT::Zero(n_blocks);
        for(Index blk : needs_reortho) {
            Index col_start = blk * this->cfg().block_size;
            auto  Qk        = Q.middleCols(col_start, this->cfg().block_size);
            for(Index prev_blk = 0; prev_blk < blk; ++prev_blk) {
                Index prev_col_start = prev_blk * this->cfg().block_size;
                auto  Qj             = Q.middleCols(prev_col_start, this->cfg().block_size);
                MatrixType proj      = Qj.adjoint() * Qk;
                Qk                  -= Qj * proj;
            }
            active_block_mask(blk) = Qk.norm() > this->normTol;
            change_block_mask(blk) = 1;
            Eigen::HouseholderQR<MatrixType> qk_hhqr(Qk);
            Qk = qk_hhqr.householderQ().setLength(Qk.cols()) * MatrixType::Identity(Q.rows(), this->cfg().block_size);
        }
        return {active_block_mask, change_block_mask};
    }

    template<typename Scalar, grit::Form form_>
    void lobpcg<Scalar, form_>::build() {
        const Eigen::Index b = this->cfg().block_size;
        const Eigen::Index N = this->N;

        MatrixType Q_prev_basis = Q;

        Eigen::Index qBlocks_old = qBlocks;
        Eigen::Index wBlocks_old = wBlocks;
        Eigen::Index mBlocks_old = mBlocks;
        Eigen::Index sBlocks_old = sBlocks;
        Eigen::Index rBlocks_old = rBlocks;

        auto get_total_blocks = [&]() { return qBlocks + wBlocks + mBlocks + sBlocks + rBlocks; };

        qBlocks = status.iter <= 1 ? 1 : 2;
        wBlocks = std::min(max_wBlocks, wBlocks_old + 1);
        mBlocks = T_evals.size() >= 2 * b ? std::min(max_mBlocks, mBlocks_old + 1) : 0;
        sBlocks = S.cols() == b ? std::min(max_sBlocks, sBlocks_old + 1) : 0;
        rBlocks = (config.inject_randomness && status.iter > 20 && status.iter % 20 == 0 && get_total_blocks() * b <= N) ? 1 : 0;

        while(N < get_total_blocks() * b) {
            if(rBlocks > 0) {
                rBlocks--;
                continue;
            }
            if(mBlocks > 0) {
                mBlocks--;
                continue;
            }
            if(wBlocks > 0) {
                wBlocks--;
                continue;
            }
            if(sBlocks > 1) {
                sBlocks--;
                continue;
            }
            break;
        }

        Q.conservativeResize(N, get_total_blocks() * b);

        if(status.iter <= 1) {
            Q.leftCols(b) = V;
        } else {
            if(qBlocks_old != qBlocks && wBlocks_old + mBlocks_old + sBlocks_old + rBlocks_old > 0) {
                shift_blocks_right(Q, qBlocks_old, qBlocks, wBlocks_old + mBlocks_old + sBlocks_old + rBlocks_old);
            }
            if(wBlocks_old != wBlocks && mBlocks_old + sBlocks_old + rBlocks_old > 0) {
                shift_blocks_right(Q, qBlocks + wBlocks_old, qBlocks + wBlocks, std::min(mBlocks_old, mBlocks) + std::min(sBlocks_old, sBlocks) + std::min(rBlocks_old, rBlocks));
            }
            if(mBlocks_old < mBlocks && sBlocks_old + rBlocks_old > 0) {
                shift_blocks_right(Q, qBlocks + wBlocks + mBlocks_old, qBlocks + wBlocks + mBlocks, std::min(sBlocks_old, sBlocks) + std::min(rBlocks_old, rBlocks));
            }
            if(sBlocks_old < sBlocks && rBlocks_old > 0) {
                shift_blocks_right(Q, qBlocks + wBlocks + mBlocks + sBlocks_old, qBlocks + wBlocks + mBlocks + sBlocks, std::min(rBlocks_old, rBlocks));
            }

            if(qBlocks == 2) Q.middleCols(0, b) = Q.middleCols(b, b);
            Q.middleCols((qBlocks - 1) * b, b) = V;

            if(wBlocks > 0) roll_blocks_left(Q, qBlocks, wBlocks);
            if(mBlocks > 0) roll_blocks_left(Q, qBlocks + wBlocks, mBlocks);
            if(sBlocks > 0) roll_blocks_left(Q, qBlocks + wBlocks + mBlocks, sBlocks);
        }

        Eigen::Index wOffset = qBlocks;
        Eigen::Index mOffset = qBlocks + wBlocks;
        Eigen::Index sOffset = qBlocks + wBlocks + mBlocks;
        Eigen::Index rOffset = qBlocks + wBlocks + mBlocks + sBlocks;

        if(wBlocks > 0) {
            MatrixType W_block = MultA(V);
            MatrixType A_block = V.adjoint() * W_block;
            W_block.noalias() -= V * A_block;
            if(V_prev.rows() == N && V_prev.cols() == b) {
                MatrixType B_block = V_prev.adjoint() * W_block;
                W_block.noalias() -= V_prev * B_block.adjoint();
            }
            if(A.has_preconditioner_apply() && T_evals.size() >= b) {
                auto select_b = this->get_ritz_indices(this->cfg().ritz, 0, b, T_evals);
                VectorReal evals(b);
                for(Eigen::Index j = 0; j < b; ++j) evals(j) = T_evals(select_b[static_cast<size_t>(j)]);
                W_block = MultP(W_block, evals);
            }
            Q.middleCols(wOffset * b, b) = W_block;
        }

        if(mBlocks > 0 && T_evals.size() >= 2 * b && Q_prev_basis.cols() == T_evecs.rows()) {
            auto top_2b_indices = this->get_ritz_indices(this->cfg().ritz, b, b, T_evals);
            MatrixType Z        = T_evecs(Eigen::placeholders::all, top_2b_indices);
            Q.middleCols(mOffset * b, b) = Q_prev_basis * Z;
        }

        if(sBlocks > 0 && S.cols() == b) Q.middleCols(sOffset * b, b) = S;
        if(rBlocks > 0) Q.middleCols(rOffset * b, b) = Eigen::MatrixXf::Random(N, b).template cast<Scalar>();

        AQ = MultA(Q);
        if constexpr(form_ == grit::Form::GENERALIZED) {
            BQ = MultB(Q);
            if(config.use_b_inner_product) {
                if(qBlocks * b > 0) {
                    MatrixType QL  = Q.leftCols(qBlocks * b);
                    MatrixType AQL = AQ.leftCols(qBlocks * b);
                    MatrixType BQL = BQ.leftCols(qBlocks * b);
                    typename Base::OrthMeta mQL;
                    mQL.maskPolicy = Base::MaskPolicy::COMPRESS;
                    block_bm_orthonormalize(QL, AQL, BQL, mQL);
                    if(Q.cols() > QL.cols()) {
                        MatrixType QR  = Q.rightCols(Q.cols() - QL.cols());
                        MatrixType AQR = AQ.rightCols(AQ.cols() - AQL.cols());
                        MatrixType BQR = BQ.rightCols(BQ.cols() - BQL.cols());
                        typename Base::OrthMeta mQR;
                        mQR.maskPolicy = Base::MaskPolicy::COMPRESS;
                        block_bm_orthogonalize(QL, AQL, BQL, QR, AQR, BQR, mQR);
                        Q.resize(Eigen::NoChange, QL.cols() + QR.cols());
                        AQ.resize(Eigen::NoChange, AQL.cols() + AQR.cols());
                        BQ.resize(Eigen::NoChange, BQL.cols() + BQR.cols());
                        Q.leftCols(QL.cols())   = QL;
                        AQ.leftCols(AQL.cols()) = AQL;
                        BQ.leftCols(BQL.cols()) = BQL;
                        Q.rightCols(QR.cols())  = QR;
                        AQ.rightCols(AQR.cols()) = AQR;
                        BQ.rightCols(BQR.cols()) = BQR;
                    } else {
                        Q  = QL;
                        AQ = AQL;
                        BQ = BQL;
                    }
                }
                typename Base::OrthMeta meta;
                meta.maskPolicy = Base::MaskPolicy::COMPRESS;
                block_bm_orthonormalize(Q, AQ, BQ, meta);
            } else {
                typename Base::OrthMeta meta;
                meta.maskPolicy = Base::MaskPolicy::COMPRESS;
                block_l2_orthonormalize(Q, AQ, BQ, meta);
            }
        } else {
            if(qBlocks * b > 0) {
                MatrixType QL  = Q.leftCols(qBlocks * b);
                MatrixType AQL = AQ.leftCols(qBlocks * b);
                typename Base::OrthMeta mQL;
                mQL.maskPolicy = Base::MaskPolicy::COMPRESS;
                block_l2_orthonormalize(QL, AQL, mQL);
                if(Q.cols() > QL.cols()) {
                    MatrixType QR  = Q.rightCols(Q.cols() - QL.cols());
                    MatrixType AQR = AQ.rightCols(AQ.cols() - AQL.cols());
                    typename Base::OrthMeta mQR;
                    mQR.maskPolicy = Base::MaskPolicy::COMPRESS;
                    block_l2_orthogonalize(QL, AQL, QR, AQR, mQR);
                    Q.resize(Eigen::NoChange, QL.cols() + QR.cols());
                    AQ.resize(Eigen::NoChange, AQL.cols() + AQR.cols());
                    Q.leftCols(QL.cols())   = QL;
                    AQ.leftCols(AQL.cols()) = AQL;
                    Q.rightCols(QR.cols())  = QR;
                    AQ.rightCols(AQR.cols()) = AQR;
                } else {
                    Q  = QL;
                    AQ = AQL;
                }
            }
            typename Base::OrthMeta meta;
            meta.maskPolicy = Base::MaskPolicy::COMPRESS;
            block_l2_orthonormalize(Q, AQ, meta);
            BQ = Q;
        }

        qBlocks = std::max<Eigen::Index>(1, Q.cols() / b);
        auto [active_mask, change_mask] = selective_orthonormalize();
        static_cast<void>(active_mask);
        static_cast<void>(change_mask);
    }

    template<typename Scalar, grit::Form form_>
    void lobpcg<Scalar, form_>::extractRitzVectors() {
        V_prev = V;
        Base::extractRitzVectors();
    }

    template<typename Scalar, grit::Form form_>
    void lobpcg<Scalar, form_>::run_user_callback() {
        if(config.user_callback) config.user_callback(*this);
    }
}

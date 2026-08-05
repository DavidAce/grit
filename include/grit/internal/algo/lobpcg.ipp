#pragma once

#include "grit/algo/lobpcg.h"
#include <algorithm>
#include <stdexcept>

namespace grit::algo {
    template<typename Scalar, grit::Form form_>
    lobpcg<Scalar, form_>::lobpcg(Matvec<Scalar> &A) requires(form_ == grit::Form::STANDARD)
        : Base(MatrixType{}, A) {
        this->bind_config(config);
        config.nev                       = 1;
        config.block_size                = 1;
        config.ncv                       = std::min<Eigen::Index>(8, std::max<Eigen::Index>(1, this->N));
        config.max_extra_ritz_history    = 1;
        config.max_ritz_residual_history = 1;
    }

    template<typename Scalar, grit::Form form_>
    lobpcg<Scalar, form_>::lobpcg(Matvec<Scalar> &A, Matvec<Scalar> &B) requires(form_ == grit::Form::GENERALIZED)
        : Base(MatrixType{}, A, B) {
        this->bind_config(config);
        config.nev                       = 1;
        config.block_size                = 1;
        config.ncv                       = std::min<Eigen::Index>(8, std::max<Eigen::Index>(1, this->N));
        config.max_extra_ritz_history    = 1;
        config.max_ritz_residual_history = 1;
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
        if(config.nev > config.block_size) throw std::runtime_error("lobpcg config error: nev must not exceed block_size");
        if(config.ncv < config.nev) throw std::runtime_error("lobpcg config error: ncv must be at least nev");
        if(config.ncv > this->N) throw std::runtime_error("lobpcg config error: ncv must not exceed the operator size");
        if(config.block_size > config.ncv) throw std::runtime_error("lobpcg config error: block_size must not exceed ncv");
        if(config.ncv % config.block_size != 0) throw std::runtime_error("lobpcg config error: ncv must be divisible by block_size");
        if(config.max_extra_ritz_history < 0) throw std::runtime_error("lobpcg config error: max_extra_ritz_history must be nonnegative");
        if(config.max_ritz_residual_history < 0) throw std::runtime_error("lobpcg config error: max_ritz_residual_history must be nonnegative");
        if(config.max_iters == 0) throw std::runtime_error("lobpcg config error: max_iters must be positive or negative for unlimited");
        if(config.max_matvecs == 0) throw std::runtime_error("lobpcg config error: max_matvecs must be positive or negative for unlimited");
        if(config.abstol <= RealScalar{0}) throw std::runtime_error("lobpcg config error: abstol must be positive");
        if(config.reltol < RealScalar{0}) throw std::runtime_error("lobpcg config error: reltol must be nonnegative");
        if(!std::isfinite(config.ritz_stabilization_tolerance) || config.ritz_stabilization_tolerance <= RealScalar{0}) {
            throw std::runtime_error("lobpcg config error: ritz_stabilization_tolerance must be finite and positive");
        }
        if(this->has_initial_guess()) {
            if(this->initial_guess().rows() != this->N) throw std::runtime_error("lobpcg config error: initial guess row count must match the operator size");
            if(this->initial_guess().cols() < 1) throw std::runtime_error("lobpcg config error: initial guess must have at least one column");
        }
    }

    template<typename Scalar, grit::Form form_>
    void lobpcg<Scalar, form_>::run() {
        assert_config();

        this->setLogger(config.log_level, std::string("grit|") + std::string(this->form_name()));
        Base::run();
    }

    template<typename Scalar, grit::Form form_>
    void lobpcg<Scalar, form_>::shift_blocks_right(Eigen::Ref<MatrixType> matrix, Eigen::Index offset_old, Eigen::Index offset_new, Eigen::Index extent) {
        if(extent <= 0 || offset_old == offset_new) return;
        const auto b          = this->cfg().block_size;
        const auto max_blocks = matrix.cols() / b;
        extent                = std::min({extent, max_blocks - offset_old, max_blocks - offset_new});
        if(extent <= 0) return;
        auto from = matrix.middleCols(offset_old * b, extent * b);
        auto to   = matrix.middleCols(offset_new * b, extent * b);
        to        = from.eval();
    }

    template<typename Scalar, grit::Form form_>
    void lobpcg<Scalar, form_>::roll_blocks_left(Eigen::Ref<MatrixType> matrix, Eigen::Index offset, Eigen::Index extent) {
        const auto b          = this->cfg().block_size;
        const auto max_blocks = matrix.cols() / b;
        extent                = std::min(extent, max_blocks - offset);
        if(extent <= 1) return;
        for(Eigen::Index k = extent - 1; k > 0; --k) {
            auto K0 = matrix.middleCols((offset + k + 0) * b, b);
            auto K1 = matrix.middleCols((offset + k - 1) * b, b);
            K0      = K1;
        }
    }

    template<typename Scalar, grit::Form form_>
    std::pair<typename lobpcg<Scalar, form_>::VectorIdxT, typename lobpcg<Scalar, form_>::VectorIdxT> lobpcg<Scalar, form_>::selective_orthonormalize() {
        using Index    = Eigen::Index;
        Index n_blocks = Q.cols() / this->cfg().block_size;

        if constexpr(form_ == grit::Form::GENERALIZED) {
            if(this->cfg().use_b_inner_product) return {VectorIdxT::Ones(n_blocks), VectorIdxT::Zero(n_blocks)};
        }

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
                Index      prev_col_start  = prev_blk * this->cfg().block_size;
                auto       Qj              = Q.middleCols(prev_col_start, this->cfg().block_size);
                MatrixType proj            = Qj.adjoint() * Qk;
                Qk                        -= Qj * proj;
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

        x_blocks = 1;
        w_blocks = S.cols() == b ? 1 : 0;
        p_blocks = P.cols() == b && AP.cols() == b ? 1 : 0;

        W.resize(0, 0);
        AW.resize(0, 0);
        BW.resize(0, 0);
        if(w_blocks == 1) {
            W = S;
            if(A.has_preconditioner_apply() && T_evals.size() >= b && status.optIdx.size() >= static_cast<size_t>(b)) {
                VectorReal evals(b);
                for(Eigen::Index j = 0; j < b; ++j) evals(j) = T_evals(status.optIdx.at(static_cast<size_t>(j)));
                W = MultP(W, evals);
            }
            AW = MultA(W);
            if constexpr(form_ == grit::Form::GENERALIZED) BW = MultB(W);
        }

        Q.resize(N, (x_blocks + w_blocks + p_blocks) * b);
        AQ.resize(N, Q.cols());

        Eigen::Index offset       = 0;
        Q.middleCols(offset, b)   = V;
        AQ.middleCols(offset, b)  = AV;
        offset                   += b;

        if(w_blocks == 1) {
            Q.middleCols(offset, b)   = W;
            AQ.middleCols(offset, b)  = AW;
            offset                   += b;
        }

        if(p_blocks == 1) {
            Q.middleCols(offset, b)  = P;
            AQ.middleCols(offset, b) = AP;
        }

        if constexpr(form_ == grit::Form::GENERALIZED) {
            offset = 0;
            BQ.resize(N, Q.cols());
            BQ.middleCols(offset, b)  = BV;
            offset                   += b;
            if(w_blocks == 1) {
                BQ.middleCols(offset, b)  = BW;
                offset                   += b;
            }
            if(p_blocks == 1) BQ.middleCols(offset, b) = BP;

            if(config.use_b_inner_product) {
                MatrixType              QL  = Q.leftCols(b);
                MatrixType              AQL = AQ.leftCols(b);
                MatrixType              BQL = BQ.leftCols(b);
                typename Base::OrthMeta mQL;
                mQL.maskPolicy = Base::MaskPolicy::COMPRESS;
                block_bm_orthonormalize(QL, AQL, BQL, mQL);
                if(Q.cols() > QL.cols()) {
                    MatrixType              QR  = Q.rightCols(Q.cols() - QL.cols());
                    MatrixType              AQR = AQ.rightCols(AQ.cols() - AQL.cols());
                    MatrixType              BQR = BQ.rightCols(BQ.cols() - BQL.cols());
                    typename Base::OrthMeta mQR;
                    mQR.maskPolicy = Base::MaskPolicy::COMPRESS;
                    block_bm_orthogonalize(QL, AQL, BQL, QR, AQR, BQR, mQR);
                    Q.resize(Eigen::NoChange, QL.cols() + QR.cols());
                    AQ.resize(Eigen::NoChange, AQL.cols() + AQR.cols());
                    BQ.resize(Eigen::NoChange, BQL.cols() + BQR.cols());
                    Q.leftCols(QL.cols())    = QL;
                    AQ.leftCols(AQL.cols())  = AQL;
                    BQ.leftCols(BQL.cols())  = BQL;
                    Q.rightCols(QR.cols())   = QR;
                    AQ.rightCols(AQR.cols()) = AQR;
                    BQ.rightCols(BQR.cols()) = BQR;
                } else {
                    Q  = QL;
                    AQ = AQL;
                    BQ = BQL;
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
            MatrixType              QL  = Q.leftCols(b);
            MatrixType              AQL = AQ.leftCols(b);
            typename Base::OrthMeta mQL;
            mQL.maskPolicy = Base::MaskPolicy::COMPRESS;
            block_l2_orthonormalize(QL, AQL, mQL);
            if(Q.cols() > QL.cols()) {
                MatrixType              QR  = Q.rightCols(Q.cols() - QL.cols());
                MatrixType              AQR = AQ.rightCols(AQ.cols() - AQL.cols());
                typename Base::OrthMeta mQR;
                mQR.maskPolicy = Base::MaskPolicy::COMPRESS;
                block_l2_orthogonalize(QL, AQL, QR, AQR, mQR);
                Q.resize(Eigen::NoChange, QL.cols() + QR.cols());
                AQ.resize(Eigen::NoChange, AQL.cols() + AQR.cols());
                Q.leftCols(QL.cols())    = QL;
                AQ.leftCols(AQL.cols())  = AQL;
                Q.rightCols(QR.cols())   = QR;
                AQ.rightCols(AQR.cols()) = AQR;
            } else {
                Q  = QL;
                AQ = AQL;
            }
            typename Base::OrthMeta meta;
            meta.maskPolicy = Base::MaskPolicy::COMPRESS;
            block_l2_orthonormalize(Q, AQ, meta);
            BQ = Q;
        }

        const Eigen::Index full_cols = (Q.cols() / b) * b;
        if(full_cols != Q.cols()) {
            Q.conservativeResize(Eigen::NoChange, full_cols);
            AQ.conservativeResize(Eigen::NoChange, full_cols);
            BQ.conservativeResize(Eigen::NoChange, full_cols);
        }
        qBlocks = Q.cols() / b;
        if(qBlocks < 1) throw std::runtime_error("lobpcg build error: basis lost all complete blocks");
        this->update_condition_numbers();
    }

    template<typename Scalar, grit::Form form_>
    void lobpcg<Scalar, form_>::extractRitzVectors() {
        V_prev = V;
        if(status.stopReason != StopReason::none) return;
        if(T_evals.size() < cfg().block_size) return;

        Eigen::Index k     = std::min(cfg().block_size, T_evals.size());
        Eigen::Index nritz = std::max({cfg().nev, cfg().block_size, k});

        status.optIdx = this->get_ritz_indices(cfg().ritz, 0, nritz, T_evals);
        MatrixType Z  = T_evecs(Eigen::placeholders::all, status.optIdx);

        const Eigen::Index x_cols      = cfg().block_size;
        const Eigen::Index search_cols = Q.cols() - x_cols;

        P.resize(0, 0);
        AP.resize(0, 0);
        BP.resize(0, 0);

        if(search_cols > 0 && Z.cols() >= k) {
            MatrixType Z_search = Z.bottomRows(search_cols).leftCols(k);
            P.resize(this->N, k);
            AP.resize(this->N, k);
            P.noalias()  = Q.rightCols(search_cols) * Z_search;
            AP.noalias() = AQ.rightCols(search_cols) * Z_search;

            if constexpr(form_ == grit::Form::GENERALIZED) {
                BP.resize(this->N, k);
                BP.noalias() = BQ.rightCols(search_cols) * Z_search;
            }
        }

        Base::extractRitzVectors();
    }

    template<typename Scalar, grit::Form form_>
    void lobpcg<Scalar, form_>::run_user_callback() {
        if(config.user_callback) config.user_callback(*this);
    }
}

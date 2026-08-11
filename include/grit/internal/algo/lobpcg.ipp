#pragma once

#include "grit/algo/lobpcg.h"
#include <algorithm>
#include <stdexcept>

namespace grit::algo {
    template<typename Scalar, grit::Form form_>
    lobpcg<Scalar, form_>::lobpcg(Matvec<Scalar> &A) requires(form_ == grit::Form::STANDARD)
        : Base(MatrixType{}, A) {
        config.nev        = 1;
        config.block_size = 1;
        config.ncv        = std::min<Eigen::Index>(8, std::max<Eigen::Index>(1, this->N));
    }

    template<typename Scalar, grit::Form form_>
    lobpcg<Scalar, form_>::lobpcg(Matvec<Scalar> &A, Matvec<Scalar> &B) requires(form_ == grit::Form::GENERALIZED)
        : Base(MatrixType{}, A, B) {
        config.nev        = 1;
        config.block_size = 1;
        config.ncv        = std::min<Eigen::Index>(8, std::max<Eigen::Index>(1, this->N));
    }

    template<typename Scalar, grit::Form form_> void lobpcg<Scalar, form_>::assert_config() const {
        this->assert_base_config();
        if(config.nev > config.block_size) throw std::runtime_error("lobpcg config error: nev must not exceed block_size");
    }

    template<typename Scalar, grit::Form form_> void lobpcg<Scalar, form_>::run() {
        assert_config();

        this->setLogger(config.log_level, std::string("grit|") + std::string(this->form_name()));
        Base::run();
    }

    template<typename Scalar, grit::Form form_> void lobpcg<Scalar, form_>::build() {
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
        if(Q.cols() < b) throw std::runtime_error("lobpcg build error: basis lost all complete blocks");
        this->update_condition_numbers();
    }

    template<typename Scalar, grit::Form form_> void lobpcg<Scalar, form_>::extractRitzVectors() {
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

    template<typename Scalar, grit::Form form_> void lobpcg<Scalar, form_>::run_user_callback() {
        if(config.user_callback) config.user_callback(*this);
    }
}

#pragma once

#include <algorithm>
#include <grit/algo/gdplusk.h>
#include <optional>
#include <string>

namespace grit::algo {
    template<typename Form>
    gdplusk<Form>::gdplusk(const grit::standard::problem<Scalar> &problem, const gdplusk_config<Scalar> &cfg)
        requires std::is_same_v<Form, grit::form::standard<Scalar>>
        : Form(cfg.nev, cfg.ncv, cfg.ritz, cfg.initial_guess(), problem.get_A(), cfg.logLevel), config(cfg) {
        initialize_config();
    }

    template<typename Form>
    gdplusk<Form>::gdplusk(const grit::generalized::problem<Scalar> &problem, const gdplusk_config<Scalar> &cfg)
        requires std::is_same_v<Form, grit::form::generalized<Scalar>>
        : Form(cfg.nev, cfg.ncv, cfg.ritz, cfg.initial_guess(), problem.get_A(), problem.get_B(), cfg.logLevel), config(cfg) {
        initialize_config();
    }

    template<typename Form>
    void gdplusk<Form>::initialize_config() {
        config.b                      = std::clamp<Eigen::Index>(config.b, 1, std::max<Eigen::Index>(1, this->N));
        config.nev                    = std::min<Eigen::Index>(std::max<Eigen::Index>(1, config.nev), this->N);
        config.ncv                    = std::min<Eigen::Index>(std::max<Eigen::Index>(config.nev, config.ncv), this->N);
        config.maxExtraRitzHistory    = std::clamp<Eigen::Index>(config.maxExtraRitzHistory, 0, this->N / std::max<Eigen::Index>(1, config.b));
        config.maxRitzResidualHistory = std::clamp<Eigen::Index>(config.maxRitzResidualHistory, 0, this->N / std::max<Eigen::Index>(1, config.b));
        config.maxBasisBlocks         = std::clamp<Eigen::Index>(config.maxBasisBlocks, 1, this->N / std::max<Eigen::Index>(1, config.b));
        config.maxRetainBlocks        = std::clamp<Eigen::Index>(config.maxRetainBlocks, 1, this->N / std::max<Eigen::Index>(1, config.b));
        this->b                       = config.b;
        this->nev                     = config.nev;
        this->ncv                     = config.ncv;
        this->ritz                    = config.ritz;
        this->max_iters               = config.max_iters;
        this->max_matvecs             = config.max_matvecs;
        this->abstol                  = config.abstol;
        this->reltol                  = config.reltol;
        max_mBlocks                   = config.maxExtraRitzHistory;
        max_sBlocks                   = config.maxRitzResidualHistory;
        this->setLogger(config.logLevel, std::string("grit|") + std::string(this->form_name()));
    }

    template<typename Form>
    void gdplusk<Form>::shift_blocks_right(Eigen::Ref<MatrixType> matrix, Eigen::Index offset_old, Eigen::Index offset_new, Eigen::Index extent) {
        if(extent <= 0 || offset_old == offset_new) return;
        matrix.middleCols(offset_new * b, extent * b) = matrix.middleCols(offset_old * b, extent * b);
    }

    template<typename Form>
    void gdplusk<Form>::roll_blocks_left(Eigen::Ref<MatrixType> matrix, Eigen::Index offset, Eigen::Index extent) {
        if(extent <= 1) return;
        matrix.middleCols(offset * b, (extent - 1) * b) = matrix.middleCols((offset + 1) * b, (extent - 1) * b);
    }

    template<typename Form>
    void gdplusk<Form>::selective_orthonormalize(const Eigen::Ref<const MatrixType> X, Eigen::Ref<MatrixType> Y, RealScalar breakdownTol, VectorIdxT &mask) {
        for(Eigen::Index j = 0; j < Y.cols(); ++j) {
            if(X.cols() > 0) Y.col(j).noalias() -= X * (X.adjoint() * Y.col(j));
            auto norm = Y.col(j).norm();
            mask(j)   = norm > breakdownTol ? 1 : 0;
            if(mask(j)) Y.col(j) /= norm;
        }
    }

    template<typename Form>
    void gdplusk<Form>::make_new_Q_block(fMultP_t fMultP) {
        if(S.cols() == 0) return;
        VectorReal evals = status.eigVal;
        Q_new            = use_preconditioner ? fMultP(S, evals, std::nullopt) : S;
        MatrixType basis = Q.cols() == 0 ? V : Q;
        if(basis.cols() > 0) Q_new.noalias() -= basis * (basis.adjoint() * Q_new);
        OrthMeta   m;
        MatrixType AQ_tmp = MultA(Q_new);
        MatrixType BQ_tmp = MultB(Q_new);
        block_l2_orthonormalize(Q_new, AQ_tmp, BQ_tmp, m);
        AQ_new = AQ_tmp;
        BQ_new = BQ_tmp;
    }

    template<typename Form>
    void gdplusk<Form>::build() {
        auto fMultP = [this](const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const VectorReal> &evals,
                             std::optional<const Eigen::Ref<const MatrixType>> iG) -> MatrixType { return this->MultP(X, evals, iG); };
        make_new_Q_block(fMultP);
        if(Q_new.cols() == 0) return;
        if(Q.cols() + Q_new.cols() > ncv) {
            Q  = V;
            AQ = AV;
            BQ = BV;
        }
        build(Q, AQ, BQ, Q_new, AQ_new, BQ_new);
    }

    template<typename Form>
    void gdplusk<Form>::build(MatrixType &Q, MatrixType &AQ, const MatrixType &Q_new, const MatrixType &AQ_new) {
        MatrixType BQ_id = Q;
        build(Q, AQ, BQ_id, Q_new, AQ_new, Q_new);
    }

    template<typename Form>
    void gdplusk<Form>::build(MatrixType &Q, MatrixType &AQ, MatrixType &BQ, const MatrixType &Q_new, const MatrixType &AQ_new, const MatrixType &BQ_new) {
        auto old_cols = Q.cols();
        Q.conservativeResize(Eigen::NoChange, old_cols + Q_new.cols());
        AQ.conservativeResize(Eigen::NoChange, old_cols + AQ_new.cols());
        BQ.conservativeResize(Eigen::NoChange, old_cols + BQ_new.cols());
        Q.rightCols(Q_new.cols())   = Q_new;
        AQ.rightCols(AQ_new.cols()) = AQ_new;
        BQ.rightCols(BQ_new.cols()) = BQ_new;
        qBlocks                     = Q.cols() / std::max<Eigen::Index>(1, b);
    }

    template<typename Form>
    void gdplusk<Form>::set_maxExtraRitzHistory(Eigen::Index m) {
        config.maxExtraRitzHistory = std::clamp<Eigen::Index>(m, 0, N / std::max<Eigen::Index>(1, b));
        max_mBlocks                = config.maxExtraRitzHistory;
    }
    template<typename Form>
    void gdplusk<Form>::set_maxRitzResidualHistory(Eigen::Index s) {
        config.maxRitzResidualHistory = std::clamp<Eigen::Index>(s, 0, N / std::max<Eigen::Index>(1, b));
        max_sBlocks                   = config.maxRitzResidualHistory;
    }
    template<typename Form>
    void gdplusk<Form>::set_maxBasisBlocks(Eigen::Index bb) {
        config.maxBasisBlocks = std::clamp<Eigen::Index>(bb, 1, N / std::max<Eigen::Index>(1, b));
        ncv                   = std::min<Eigen::Index>(N, config.maxBasisBlocks * b);
    }
    template<typename Form>
    void gdplusk<Form>::set_maxRetainBlocks(Eigen::Index rb) {
        config.maxRetainBlocks = std::clamp<Eigen::Index>(rb, 1, N / std::max<Eigen::Index>(1, b));
    }
}

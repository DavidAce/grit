#pragma once

#include <algorithm>
#include <grit/algo/gdplusk.h>
#include <optional>
#include <stdexcept>
#include <string>

namespace grit::algo {
    template<typename Form>
    gdplusk<Form>::gdplusk(const grit::standard::problem<Scalar> &problem, const gdplusk_config<Scalar> &cfg)
        requires std::is_same_v<Form, grit::form::standard<Scalar>>
        : Form(cfg.nev, cfg.ncv, cfg.ritz, cfg.initial_guess(), problem.get_A(), cfg.log_level), config(cfg) {
        initialize_config();
    }

    template<typename Form>
    gdplusk<Form>::gdplusk(const grit::generalized::problem<Scalar> &problem, const gdplusk_config<Scalar> &cfg)
        requires std::is_same_v<Form, grit::form::generalized<Scalar>>
        : Form(cfg.nev, cfg.ncv, cfg.ritz, cfg.initial_guess(), problem.get_A(), problem.get_B(), cfg.log_level), config(cfg) {
        initialize_config();
    }

    template<typename Form>
    void gdplusk<Form>::initialize_config() {
        config.nev                    = std::min<Eigen::Index>(std::max<Eigen::Index>(1, config.nev), this->N);
        config.block_size             = std::clamp<Eigen::Index>(std::max<Eigen::Index>(config.nev, config.block_size), 1, std::max<Eigen::Index>(1, this->N));
        const auto max_basis_blocks   = std::max<Eigen::Index>(1, this->N / std::max<Eigen::Index>(1, config.block_size));
        config.max_extra_ritz_history = std::clamp<Eigen::Index>(config.max_extra_ritz_history, 0, max_basis_blocks);
        config.max_ritz_residual_history = std::clamp<Eigen::Index>(config.max_ritz_residual_history, 0, max_basis_blocks);
        if(config.ncv < 0) {
            config.max_basis_blocks = std::clamp<Eigen::Index>(config.max_basis_blocks, 1, max_basis_blocks);
            config.ncv              = config.max_basis_blocks * config.block_size;
        } else {
            config.ncv              = std::min<Eigen::Index>(std::max<Eigen::Index>(config.nev, config.ncv), this->N);
            config.max_basis_blocks = std::clamp<Eigen::Index>(config.ncv / config.block_size, 1, max_basis_blocks);
            config.ncv              = config.max_basis_blocks * config.block_size;
        }
        config.max_retain_blocks            = std::clamp<Eigen::Index>(config.max_retain_blocks, 1, config.max_basis_blocks);
        this->block_size                    = config.block_size;
        this->nev                           = config.nev;
        this->ncv                           = config.ncv;
        this->ritz                          = config.ritz;
        this->max_iters                     = config.max_iters;
        this->max_matvecs                   = config.max_matvecs;
        this->tol                           = config.tol;
        this->tol_rnorm_relative            = config.tol_rnorm_relative;
        this->sat_eigval_threshold          = config.sat_eigval_threshold;
        this->sat_rnorm_threshold           = config.sat_rnorm_threshold;
        this->inner_tol                     = config.inner_tol;
        this->inner_max_iters               = config.inner_max_iters;
        this->auto_min_dwell_iters          = std::max<Eigen::Index>(0, config.auto_min_dwell_iters);
        this->auto_sat_eigval_threshold     = std::max<RealScalar>(RealScalar{0}, config.auto_sat_eigval_threshold);
        this->auto_sat_rnorm_threshold      = std::max<RealScalar>(RealScalar{0}, config.auto_sat_rnorm_threshold);
        this->auto_jd_start_rnorm_threshold = std::max<RealScalar>(RealScalar{0}, config.auto_jd_start_rnorm_threshold);
        this->auto_cheap_probe_interval     = std::max<Eigen::Index>(1, config.auto_cheap_probe_interval);
        this->auto_cheap_probe_factor       = std::max<RealScalar>(RealScalar{0}, config.auto_cheap_probe_factor);
        this->status_callback               = config.status_callback;
        max_mBlocks                         = config.max_extra_ritz_history;
        max_sBlocks                         = config.max_ritz_residual_history;
        this->setLogger(config.log_level, std::string("grit|") + std::string(this->form_name()));
    }

    template<typename Form>
    void gdplusk<Form>::shift_blocks_right(Eigen::Ref<MatrixType> matrix, Eigen::Index offset_old, Eigen::Index offset_new, Eigen::Index extent) {
        if(extent <= 0 || offset_old == offset_new) return;
        matrix.middleCols(offset_new * block_size, extent * block_size) = matrix.middleCols(offset_old * block_size, extent * block_size);
    }

    template<typename Form>
    void gdplusk<Form>::roll_blocks_left(Eigen::Ref<MatrixType> matrix, Eigen::Index offset, Eigen::Index extent) {
        if(extent <= 1) return;
        matrix.middleCols(offset * block_size, (extent - 1) * block_size) = matrix.middleCols((offset + 1) * block_size, (extent - 1) * block_size);
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
    void gdplusk<Form>::make_new_Q_block() {
        if(S.cols() == 0) return;
        Q_new = get_sBlock(S);

        auto orthogonalize_Q_new = [&]() {
            OrthMeta m;
            m.maskPolicy = Form::MaskPolicy::COMPRESS;
            AQ_new       = MatrixType();
            BQ_new       = MatrixType();

            if(is_generalized_problem()) {
                if(use_b_inner_product) {
                    if(V.cols() > 0) block_bm_orthogonalize(V, AV, BV, Q_new, AQ_new, BQ_new, m);
                    if(Q.cols() > 0) block_bm_orthogonalize(Q, AQ, BQ, Q_new, AQ_new, BQ_new, m);
                    block_bm_orthonormalize(Q_new, AQ_new, BQ_new, m);
                } else {
                    if(V.cols() > 0) block_l2_orthogonalize(V, AV, BV, Q_new, AQ_new, BQ_new, m);
                    if(Q.cols() > 0) block_l2_orthogonalize(Q, AQ, BQ, Q_new, AQ_new, BQ_new, m);
                    block_l2_orthonormalize(Q_new, AQ_new, BQ_new, m);
                }
            } else {
                if(V.cols() > 0) block_l2_orthogonalize(V, AV, Q_new, AQ_new, m);
                if(Q.cols() > 0) block_l2_orthogonalize(Q, AQ, Q_new, AQ_new, m);
                block_l2_orthonormalize(Q_new, AQ_new, m);
                BQ_new = Q_new;
            }
        };

        orthogonalize_Q_new();
        if(Q_new.cols() == 0 && inject_randomness) {
            if(eiglog) eiglog->debug("Replacing Q_new with a random vector");
            Q_new = Eigen::MatrixXf::Random(N, block_size).template cast<Scalar>();
            orthogonalize_Q_new();
        }
    }

    template<typename Form>
    void gdplusk<Form>::build() {
        bool had_residual = S.cols() > 0;
        make_new_Q_block();
        build(Q, AQ, BQ, had_residual ? Q_new : MatrixType{}, had_residual ? AQ_new : MatrixType{}, had_residual ? BQ_new : MatrixType{});
    }

    template<typename Form>
    void gdplusk<Form>::build(MatrixType &Q, MatrixType &AQ, const MatrixType &Q_new, const MatrixType &AQ_new) {
        MatrixType BQ_id = Q;
        build(Q, AQ, BQ_id, Q_new, AQ_new, Q_new);
    }

    template<typename Form>
    void gdplusk<Form>::build(MatrixType &Q, MatrixType &AQ, MatrixType &BQ, const MatrixType &Q_new, const MatrixType &AQ_new, const MatrixType &BQ_new) {
        if(status.stopReason != StopReason::none) return;

        if(Q_new.cols() == 0 && status.iter <= status.iter_last_restart + 2) {
            status.stopReason |= StopReason::subspace_exhausted;
            status.stopMessage.emplace_back(fmt::format("saturated basis: exhausted subspace search | iter {} | mv {} | {:.3e} s", status.iter,
                                                        status.num_matvecs_total, status.time_elapsed.get_time()));
            return;
        }
        if(Q_new.cols() == 0) return;

        auto restart_basis = [&]() {
            Eigen::Index cols_ks = 0;

            if(is_generalized_problem()) {
                MatrixType T1 = Q.adjoint() * AQ;
                MatrixType T2 = Q.adjoint() * BQ;
                T1            = (T1 + T1.adjoint()) * Form::half;
                T2            = (T2 + T2.adjoint()) * Form::half;

                auto [W, Winv] = get_bm_normalizer_for_the_projected_pencil(T2);
                cols_ks        = std::clamp(std::min(config.max_retain_blocks * block_size, W.cols()), block_size, W.cols());

                MatrixType WT1W = W.adjoint() * T1 * W;
                MatrixType WT2W = W.adjoint() * T2 * W;
                WT1W            = (WT1W + WT1W.adjoint()) * Form::half;
                WT2W            = (WT2W + WT2W.adjoint()) * Form::half;

                Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType> ges(WT1W, WT2W, Eigen::Ax_lBx);
                if(ges.info() != Eigen::Success) throw std::runtime_error("gdplusk restart: generalized eigensolver failed");
                cols_ks        = std::min(cols_ks, ges.eigenvalues().size());
                auto selectIdx = this->get_ritz_indices(ritz, 0, cols_ks, ges.eigenvalues());

                VectorReal Y     = ges.eigenvalues()(selectIdx);
                MatrixType Z_rr  = ges.eigenvectors()(Eigen::placeholders::all, selectIdx);
                MatrixType Z_ref = get_refined_ritz_eigenvectors_gen(Z_rr, Y, AQ, BQ);
                MatrixType Z_opt = get_optimal_rayleigh_ritz_matrix(Z_rr, Z_ref, WT1W, WT2W);
                MatrixType Z     = W * Z_opt;
                orthonormalize_Z(Z, T2);

                MatrixType Q_ks  = Q * Z;
                MatrixType AQ_ks = AQ * Z;
                MatrixType BQ_ks = BQ * Z;

                MatrixType AK_prev, BK_prev;
                {
                    OrthMeta m;
                    m.maskPolicy = Form::MaskPolicy::COMPRESS;
                    if(use_b_inner_product) {
                        block_bm_orthonormalize(Q_ks, AQ_ks, BQ_ks, m);
                        block_bm_orthogonalize(Q_ks, AQ_ks, BQ_ks, K_prev, AK_prev, BK_prev, m);
                        block_bm_orthonormalize(K_prev, AK_prev, BK_prev, m);
                    } else {
                        block_l2_orthonormalize(Q_ks, AQ_ks, BQ_ks, m);
                        block_l2_orthogonalize(Q_ks, AQ_ks, BQ_ks, K_prev, AK_prev, BK_prev, m);
                        block_l2_orthonormalize(K_prev, AK_prev, BK_prev, m);
                    }
                }

                Q.conservativeResize(N, Q_ks.cols() + K_prev.cols());
                if(Q_ks.cols() > 0) Q.leftCols(Q_ks.cols()) = Q_ks;
                if(K_prev.cols() > 0) Q.rightCols(K_prev.cols()) = K_prev;

                AQ.conservativeResize(N, AQ_ks.cols() + AK_prev.cols());
                if(AQ_ks.cols() > 0) AQ.leftCols(AQ_ks.cols()) = AQ_ks;
                if(AK_prev.cols() > 0) AQ.rightCols(AK_prev.cols()) = AK_prev;

                BQ.conservativeResize(N, BQ_ks.cols() + BK_prev.cols());
                if(BQ_ks.cols() > 0) BQ.leftCols(BQ_ks.cols()) = BQ_ks;
                if(BK_prev.cols() > 0) BQ.rightCols(BK_prev.cols()) = BK_prev;

                OrthMeta m;
                m.Gram       = use_b_inner_product ? Q.adjoint() * BQ : Q.adjoint() * Q;
                m.Gram       = (m.Gram + m.Gram.adjoint()).eval() * Form::half;
                m.orthError  = (m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols())).norm();
                m.maskPolicy = Form::MaskPolicy::COMPRESS;
                if(use_b_inner_product) {
                    block_bm_orthonormalize(Q, AQ, BQ, m);
                } else {
                    block_l2_orthonormalize(Q, AQ, BQ, m);
                }
            } else {
                MatrixType T = Q.adjoint() * AQ;
                T            = (T + T.adjoint()) * Form::half;
                Eigen::SelfAdjointEigenSolver<MatrixType> es(T, Eigen::ComputeEigenvectors);
                if(es.info() != Eigen::Success) throw std::runtime_error("gdplusk restart: eigensolver failed");
                cols_ks        = std::clamp(std::min(config.max_retain_blocks * block_size, Q.cols()), block_size, Q.cols());
                cols_ks        = std::min(cols_ks, es.eigenvalues().size());
                auto selectIdx = this->get_ritz_indices(ritz, 0, cols_ks, es.eigenvalues());

                VectorReal Y    = es.eigenvalues()(selectIdx);
                MatrixType Z_rr = es.eigenvectors()(Eigen::placeholders::all, selectIdx);
                MatrixType Z    = get_refined_ritz_eigenvectors_std(Z_rr, Y, Q, AQ);
                orthonormalize_Z(Z, T);

                MatrixType Q_ks  = Q * Z;
                MatrixType AQ_ks = AQ * Z;

                MatrixType AK_prev;
                {
                    OrthMeta m;
                    m.maskPolicy = Form::MaskPolicy::COMPRESS;
                    block_l2_orthogonalize(Q_ks, AQ_ks, K_prev, AK_prev, m);
                    block_l2_orthonormalize(K_prev, AK_prev, m);
                }

                Q.conservativeResize(N, Q_ks.cols() + K_prev.cols());
                if(Q_ks.cols() > 0) Q.leftCols(Q_ks.cols()) = Q_ks;
                if(K_prev.cols() > 0) Q.rightCols(K_prev.cols()) = K_prev;

                AQ.conservativeResize(N, AQ_ks.cols() + AK_prev.cols());
                if(AQ_ks.cols() > 0) AQ.leftCols(AQ_ks.cols()) = AQ_ks;
                if(AK_prev.cols() > 0) AQ.rightCols(AK_prev.cols()) = AK_prev;

                BQ = Q;

                OrthMeta m;
                m.Gram       = Q.adjoint() * Q;
                m.Gram       = (m.Gram + m.Gram.adjoint()).eval() * Form::half;
                m.orthError  = (m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols())).norm();
                m.maskPolicy = Form::MaskPolicy::COMPRESS;
                block_l2_orthonormalize(Q, AQ, m);
                BQ = Q;
            }

            status.iter_last_restart = status.iter;
        };

        auto newCols = std::min<Eigen::Index>({Q.cols() + Q_new.cols(), N});
        if(newCols > config.max_basis_blocks * block_size || Q_new.cols() == 0) restart_basis();
        if(Q_new.cols() == 0) return;

        assert(Q_new.rows() == N);

        auto availableCols = std::max<Eigen::Index>(0, N - Q.cols());
        auto copyCols      = std::min<Eigen::Index>(availableCols, Q_new.cols());
        if(copyCols == 0) return;

        newCols = Q.cols() + copyCols;

        Q.conservativeResize(N, newCols);
        AQ.conservativeResize(N, newCols);
        BQ.conservativeResize(N, newCols);

        Q.rightCols(copyCols)  = Q_new.leftCols(copyCols);
        AQ.rightCols(copyCols) = AQ_new.leftCols(copyCols);
        BQ.rightCols(copyCols) = BQ_new.leftCols(copyCols);

        OrthMeta m;
        m.maskPolicy = Form::MaskPolicy::COMPRESS;
        m.Gram       = Q.adjoint() * Q;
        m.Gram       = (m.Gram + m.Gram.adjoint()).eval() * Form::half;
        m.orthError  = (m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols())).norm();

        bool basis_was_restarted = status.iter_last_restart == status.iter;
        if(basis_was_restarted || m.orthError > this->normTol * std::sqrt(status.op_norm_estimate)) {
            if(is_generalized_problem()) {
                if(use_b_inner_product) {
                    block_bm_orthonormalize(Q, AQ, BQ, m);
                } else {
                    block_l2_orthonormalize(Q, AQ, BQ, m);
                }
            } else {
                block_l2_orthonormalize(Q, AQ, m);
                BQ = Q;
            }
        }
        qBlocks = Q.cols() / std::max<Eigen::Index>(1, block_size);
    }

    template<typename Form>
    void gdplusk<Form>::set_maxExtraRitzHistory(Eigen::Index m) {
        config.max_extra_ritz_history = std::clamp<Eigen::Index>(m, 0, N / std::max<Eigen::Index>(1, block_size));
        max_mBlocks                   = config.max_extra_ritz_history;
    }
    template<typename Form>
    void gdplusk<Form>::set_maxRitzResidualHistory(Eigen::Index s) {
        config.max_ritz_residual_history = std::clamp<Eigen::Index>(s, 0, N / std::max<Eigen::Index>(1, block_size));
        max_sBlocks                      = config.max_ritz_residual_history;
    }
    template<typename Form>
    void gdplusk<Form>::set_maxBasisBlocks(Eigen::Index bb) {
        config.max_basis_blocks = std::clamp<Eigen::Index>(bb, 1, N / std::max<Eigen::Index>(1, block_size));
        ncv                     = std::min<Eigen::Index>(N, config.max_basis_blocks * block_size);
    }
    template<typename Form>
    void gdplusk<Form>::set_maxRetainBlocks(Eigen::Index rb) {
        config.max_retain_blocks = std::clamp<Eigen::Index>(rb, 1, N / std::max<Eigen::Index>(1, block_size));
    }
}

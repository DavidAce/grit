#pragma once

#include <algorithm>
#include "grit/algo/gdplusk.h"
#include <optional>
#include <stdexcept>
#include <string>

namespace grit::algo {
    template<typename Scalar, grit::Form form_>
    const typename gdplusk<Scalar, form_>::MatrixType &gdplusk<Scalar, form_>::default_initial_guess() {
        static const MatrixType guess;
        return guess;
    }

    template<typename Scalar, grit::Form form_>
    gdplusk<Scalar, form_>::gdplusk(Matvec<Scalar> &A) requires(form_ == grit::Form::STANDARD)
        : Base(default_initial_guess(), A) {
        this->bind_config(config);
        config.nev              = 1;
        config.block_size       = 1;
        config.ncv              = std::min<Eigen::Index>(8, std::max<Eigen::Index>(1, this->N));
        config.max_basis_blocks = config.ncv;
    }

    template<typename Scalar, grit::Form form_>
    gdplusk<Scalar, form_>::gdplusk(Matvec<Scalar> &A, Matvec<Scalar> &B) requires(form_ == grit::Form::GENERALIZED)
        : Base(default_initial_guess(), A, B) {
        this->bind_config(config);
        config.nev              = 1;
        config.block_size       = 1;
        config.ncv              = std::min<Eigen::Index>(8, std::max<Eigen::Index>(1, this->N));
        config.max_basis_blocks = config.ncv;
    }

    template<typename Scalar, grit::Form form_>
    void gdplusk<Scalar, form_>::set_initial_guess(const MatrixType &guess) {
        initial_guess_ptr = &guess;
    }

    template<typename Scalar, grit::Form form_>
    void gdplusk<Scalar, form_>::clear_initial_guess() {
        initial_guess_ptr = nullptr;
    }

    template<typename Scalar, grit::Form form_>
    bool gdplusk<Scalar, form_>::has_initial_guess() const {
        return initial_guess_ptr != nullptr;
    }

    template<typename Scalar, grit::Form form_>
    const typename gdplusk<Scalar, form_>::MatrixType &gdplusk<Scalar, form_>::initial_guess() const {
        if(initial_guess_ptr) return *initial_guess_ptr;
        return empty_initial_guess;
    }

    template<typename Scalar, grit::Form form_>
    void gdplusk<Scalar, form_>::assert_operator_config() const requires(form_ == grit::Form::STANDARD)
    {
        if(this->A.get_size() <= 0) throw std::runtime_error("gdplusk requires operator A with positive size");
        if(config.use_b_inner_product) throw std::runtime_error("use_b_inner_product requires a generalized problem");
        if(config.use_jd_b_only) throw std::runtime_error("use_jd_b_only requires a generalized problem");
    }

    template<typename Scalar, grit::Form form_>
    void gdplusk<Scalar, form_>::assert_operator_config() const requires(form_ == grit::Form::GENERALIZED)
    {
        if(this->A.get_size() <= 0) throw std::runtime_error("gdplusk requires operator A with positive size");
        auto &B = this->B->get();
        if(B.get_size() <= 0) throw std::runtime_error("gdplusk requires operator B with positive size");
        if(this->A.get_size() != B.get_size()) throw std::runtime_error("gdplusk requires operators A and B to have matching sizes");
    }

    template<typename Scalar, grit::Form form_>
    void gdplusk<Scalar, form_>::assert_config() const {
        assert_operator_config();

        if(config.nev < 1) throw std::runtime_error("gdplusk config error: nev must be at least 1");
        if(config.block_size < 1) throw std::runtime_error("gdplusk config error: block_size must be at least 1");
        if(config.ncv < config.nev) throw std::runtime_error("gdplusk config error: ncv must be at least nev");
        if(config.ncv > this->N) throw std::runtime_error("gdplusk config error: ncv must not exceed the operator size");
        if(config.block_size > config.ncv) throw std::runtime_error("gdplusk config error: block_size must not exceed ncv");
        if(config.ncv % config.block_size != 0) throw std::runtime_error("gdplusk config error: ncv must be divisible by block_size");
        if(config.max_basis_blocks < 1) throw std::runtime_error("gdplusk config error: max_basis_blocks must be at least 1");
        if(config.max_basis_blocks * config.block_size != config.ncv) {
            throw std::runtime_error("gdplusk config error: max_basis_blocks * block_size must equal ncv");
        }
        if(config.maxRetainBlocks < 1) throw std::runtime_error("gdplusk config error: maxRetainBlocks must be at least 1");
        if(config.maxRetainBlocks > config.max_basis_blocks) {
            throw std::runtime_error("gdplusk config error: maxRetainBlocks must not exceed max_basis_blocks");
        }
        if(config.max_extra_ritz_history < 0) throw std::runtime_error("gdplusk config error: max_extra_ritz_history must be nonnegative");
        if(config.max_extra_ritz_history > config.max_basis_blocks) {
            throw std::runtime_error("gdplusk config error: max_extra_ritz_history must not exceed max_basis_blocks");
        }
        if(config.max_ritz_residual_history < 0) throw std::runtime_error("gdplusk config error: max_ritz_residual_history must be nonnegative");
        if(config.max_ritz_residual_history > config.max_basis_blocks) {
            throw std::runtime_error("gdplusk config error: max_ritz_residual_history must not exceed max_basis_blocks");
        }
        if(config.max_iters == 0) throw std::runtime_error("gdplusk config error: max_iters must be positive or negative for unlimited");
        if(config.max_matvecs == 0) throw std::runtime_error("gdplusk config error: max_matvecs must be positive or negative for unlimited");
        if(config.tol <= RealScalar{0}) throw std::runtime_error("gdplusk config error: tol must be positive");
        if(config.tol_rnorm_relative < RealScalar{0}) throw std::runtime_error("gdplusk config error: tol_rnorm_relative must be nonnegative");
        if(config.sat_eigval_threshold < RealScalar{0}) throw std::runtime_error("gdplusk config error: sat_eigval_threshold must be nonnegative");
        if(config.sat_rnorm_threshold < RealScalar{0}) throw std::runtime_error("gdplusk config error: sat_rnorm_threshold must be nonnegative");
        if(config.inner_tol <= RealScalar{0} || config.inner_tol > RealScalar{1}) {
            throw std::runtime_error("gdplusk config error: inner_tol must be in the interval (0, 1]");
        }
        if(config.inner_max_iters < 1) throw std::runtime_error("gdplusk config error: inner_max_iters must be at least 1");
        if(config.auto_min_dwell_iters < 0) throw std::runtime_error("gdplusk config error: auto_min_dwell_iters must be nonnegative");
        if(config.auto_sat_eigval_threshold < RealScalar{0}) {
            throw std::runtime_error("gdplusk config error: auto_sat_eigval_threshold must be nonnegative");
        }
        if(config.auto_sat_rnorm_threshold < RealScalar{0}) { throw std::runtime_error("gdplusk config error: auto_sat_rnorm_threshold must be nonnegative"); }
        if(config.auto_jd_start_rnorm_threshold < RealScalar{0}) {
            throw std::runtime_error("gdplusk config error: auto_jd_start_rnorm_threshold must be nonnegative");
        }
        if(config.auto_cheap_probe_interval < 1) { throw std::runtime_error("gdplusk config error: auto_cheap_probe_interval must be at least 1"); }
        if(config.auto_cheap_probe_factor < RealScalar{0}) { throw std::runtime_error("gdplusk config error: auto_cheap_probe_factor must be nonnegative"); }
        if(has_initial_guess()) {
            if(initial_guess().rows() != this->N) throw std::runtime_error("gdplusk config error: initial guess row count must match the operator size");
            if(initial_guess().cols() < 1) throw std::runtime_error("gdplusk config error: initial guess must have at least one column");
        }
    }

    template<typename Scalar, grit::Form form_>
    void gdplusk<Scalar, form_>::run() {
        assert_config();

        this->setLogger(config.log_level, std::string("grit|") + std::string(this->form_name()));
        this->V                 = initial_guess();
        this->current_inner_tol = config.inner_tol;
        max_mBlocks             = config.max_extra_ritz_history;
        max_sBlocks             = config.max_ritz_residual_history;
        clear_result();
        Base::run();
    }

    template<typename Scalar, grit::Form form_>
    void gdplusk<Scalar, form_>::preamble() {
        Base::preamble();
        adjust_inner_tolerance(S);
        adjust_residual_correction_type();
        if(config.residual_correction_type == ResidualCorrectionType::AUTO) auto_residual_correction.step_time_start = status.time_elapsed.get_time();
    }

    template<typename Scalar, grit::Form form_>
    void gdplusk<Scalar, form_>::updateStatus() {
        Base::updateStatus();
        update_auto_residual_correction_state();
    }

    template<typename Scalar, grit::Form form_>
    void gdplusk<Scalar, form_>::extractRitzVectors() {
        if(status.stopReason != StopReason::none) return;
        if(T_evals.size() < this->cfg().block_size) return;

        Eigen::Index k     = std::min(config.maxRetainBlocks * this->cfg().block_size, T_evals.size());
        Eigen::Index nritz = std::max({this->cfg().nev, this->cfg().block_size, k});

        status.optIdx = this->get_ritz_indices(this->cfg().ritz, 0, nritz, T_evals);

        if(this->cfg().use_refined_rayleigh_ritz) {
            if constexpr(form_ == grit::Form::GENERALIZED) {
                Base::refinedRitzVectors(status.optIdx, V, AV, BV, S, status.rNorms);
            } else {
                Base::refinedRitzVectors(status.optIdx, V, AV, S, status.rNorms);
                BV = V;
            }
        } else {
            if constexpr(form_ == grit::Form::GENERALIZED) {
                Base::extractRitzVectors(status.optIdx, V, AV, BV, S, status.rNorms);
            } else {
                Base::extractRitzVectors(status.optIdx, V, AV, S, status.rNorms);
                BV = V;
            }
        }

        K_prev = K;
        K      = V.leftCols(k);

        if(k > this->cfg().block_size) {
            V.conservativeResize(Eigen::NoChange, this->cfg().block_size);
            AV.conservativeResize(Eigen::NoChange, this->cfg().block_size);
            BV.conservativeResize(Eigen::NoChange, this->cfg().block_size);
            S.conservativeResize(Eigen::NoChange, this->cfg().block_size);
            status.rNorms.conservativeResize(this->cfg().block_size);
        }

        if(status.rNorms_init.size() != status.rNorms.size()) status.rNorms_init = status.rNorms;
    }

    template<typename Scalar, grit::Form form_>
    void gdplusk<Scalar, form_>::run_user_callback() {
        if(config.user_callback) config.user_callback(*this);
    }

    template<typename Scalar, grit::Form form_>
    void gdplusk<Scalar, form_>::shift_blocks_right(Eigen::Ref<MatrixType> matrix, Eigen::Index offset_old, Eigen::Index offset_new, Eigen::Index extent) {
        if(extent <= 0 || offset_old == offset_new) return;
        matrix.middleCols(offset_new * this->cfg().block_size, extent * this->cfg().block_size) =
            matrix.middleCols(offset_old * this->cfg().block_size, extent * this->cfg().block_size);
    }

    template<typename Scalar, grit::Form form_>
    void gdplusk<Scalar, form_>::roll_blocks_left(Eigen::Ref<MatrixType> matrix, Eigen::Index offset, Eigen::Index extent) {
        if(extent <= 1) return;
        matrix.middleCols(offset * this->cfg().block_size, (extent - 1) * this->cfg().block_size) =
            matrix.middleCols((offset + 1) * this->cfg().block_size, (extent - 1) * this->cfg().block_size);
    }

    template<typename Scalar, grit::Form form_>
    void gdplusk<Scalar, form_>::selective_orthonormalize(const Eigen::Ref<const MatrixType> X, Eigen::Ref<MatrixType> Y, RealScalar breakdownTol, VectorIdxT &mask) {
        for(Eigen::Index j = 0; j < Y.cols(); ++j) {
            if(X.cols() > 0) Y.col(j).noalias() -= X * (X.adjoint() * Y.col(j));
            auto norm = Y.col(j).norm();
            mask(j)   = norm > breakdownTol ? 1 : 0;
            if(mask(j)) Y.col(j) /= norm;
        }
    }

    template<typename Scalar, grit::Form form_>
    void gdplusk<Scalar, form_>::make_new_Q_block() {
        if(S.cols() == 0) return;
        Q_new = get_sBlock(S);

        auto orthogonalize_Q_new = [&]() {
            OrthMeta m;
            m.maskPolicy = Base::MaskPolicy::COMPRESS;
            AQ_new       = MatrixType();
            BQ_new       = MatrixType();

            if constexpr(form_ == grit::Form::GENERALIZED) {
                if(config.use_b_inner_product) {
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
        if(Q_new.cols() == 0 && config.inject_randomness) {
            if(eiglog) eiglog->debug("Replacing Q_new with a random vector");
            Q_new = Eigen::MatrixXf::Random(N, this->cfg().block_size).template cast<Scalar>();
            orthogonalize_Q_new();
        }
    }

    template<typename Scalar, grit::Form form_>
    void gdplusk<Scalar, form_>::build() {
        bool had_residual = S.cols() > 0;
        make_new_Q_block();
        build(Q, AQ, BQ, had_residual ? Q_new : MatrixType{}, had_residual ? AQ_new : MatrixType{}, had_residual ? BQ_new : MatrixType{});
    }

    template<typename Scalar, grit::Form form_>
    void gdplusk<Scalar, form_>::build(MatrixType &Q, MatrixType &AQ, const MatrixType &Q_new, const MatrixType &AQ_new) {
        MatrixType BQ_id = Q;
        build(Q, AQ, BQ_id, Q_new, AQ_new, Q_new);
    }

    template<typename Scalar, grit::Form form_>
    void gdplusk<Scalar, form_>::build(MatrixType &Q, MatrixType &AQ, MatrixType &BQ, const MatrixType &Q_new, const MatrixType &AQ_new, const MatrixType &BQ_new) {
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

            if constexpr(form_ == grit::Form::GENERALIZED) {
                MatrixType T1 = Q.adjoint() * AQ;
                MatrixType T2 = Q.adjoint() * BQ;
                T1            = (T1 + T1.adjoint()) * Base::half;
                T2            = (T2 + T2.adjoint()) * Base::half;

                auto [W, Winv] = get_bm_normalizer_for_the_projected_pencil(T2);
                cols_ks        = std::clamp(std::min(config.maxRetainBlocks * this->cfg().block_size, W.cols()), this->cfg().block_size, W.cols());

                MatrixType WT1W = W.adjoint() * T1 * W;
                MatrixType WT2W = W.adjoint() * T2 * W;
                WT1W            = (WT1W + WT1W.adjoint()) * Base::half;
                WT2W            = (WT2W + WT2W.adjoint()) * Base::half;

                Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType> ges(WT1W, WT2W, Eigen::Ax_lBx);
                if(ges.info() != Eigen::Success) throw std::runtime_error("gdplusk restart: generalized eigensolver failed");
                cols_ks        = std::min(cols_ks, ges.eigenvalues().size());
                auto selectIdx = this->get_ritz_indices(this->cfg().ritz, 0, cols_ks, ges.eigenvalues());

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
                    m.maskPolicy = Base::MaskPolicy::COMPRESS;
                    if(config.use_b_inner_product) {
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
                m.Gram       = config.use_b_inner_product ? Q.adjoint() * BQ : Q.adjoint() * Q;
                m.Gram       = (m.Gram + m.Gram.adjoint()).eval() * Base::half;
                m.orthError  = (m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols())).norm();
                m.maskPolicy = Base::MaskPolicy::COMPRESS;
                if(config.use_b_inner_product) {
                    block_bm_orthonormalize(Q, AQ, BQ, m);
                } else {
                    block_l2_orthonormalize(Q, AQ, BQ, m);
                }
            } else {
                MatrixType T = Q.adjoint() * AQ;
                T            = (T + T.adjoint()) * Base::half;
                Eigen::SelfAdjointEigenSolver<MatrixType> es(T, Eigen::ComputeEigenvectors);
                if(es.info() != Eigen::Success) throw std::runtime_error("gdplusk restart: eigensolver failed");
                cols_ks        = std::clamp(std::min(config.maxRetainBlocks * this->cfg().block_size, Q.cols()), this->cfg().block_size, Q.cols());
                cols_ks        = std::min(cols_ks, es.eigenvalues().size());
                auto selectIdx = this->get_ritz_indices(this->cfg().ritz, 0, cols_ks, es.eigenvalues());

                VectorReal Y    = es.eigenvalues()(selectIdx);
                MatrixType Z_rr = es.eigenvectors()(Eigen::placeholders::all, selectIdx);
                MatrixType Z    = get_refined_ritz_eigenvectors_std(Z_rr, Y, Q, AQ);
                orthonormalize_Z(Z, T);

                MatrixType Q_ks  = Q * Z;
                MatrixType AQ_ks = AQ * Z;

                MatrixType AK_prev;
                {
                    OrthMeta m;
                    m.maskPolicy = Base::MaskPolicy::COMPRESS;
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
                m.Gram       = (m.Gram + m.Gram.adjoint()).eval() * Base::half;
                m.orthError  = (m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols())).norm();
                m.maskPolicy = Base::MaskPolicy::COMPRESS;
                block_l2_orthonormalize(Q, AQ, m);
                BQ = Q;
            }

            status.iter_last_restart = status.iter;
        };

        auto newCols = std::min<Eigen::Index>({Q.cols() + Q_new.cols(), N});
        if(newCols > config.max_basis_blocks * this->cfg().block_size || Q_new.cols() == 0) restart_basis();
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
        m.maskPolicy = Base::MaskPolicy::COMPRESS;
        m.Gram       = Q.adjoint() * Q;
        m.Gram       = (m.Gram + m.Gram.adjoint()).eval() * Base::half;
        m.orthError  = (m.Gram - MatrixType::Identity(m.Gram.rows(), m.Gram.cols())).norm();

        bool basis_was_restarted = status.iter_last_restart == status.iter;
        if(basis_was_restarted || m.orthError > this->normTol * std::sqrt(status.op_norm_estimate)) {
            if constexpr(form_ == grit::Form::GENERALIZED) {
                if(config.use_b_inner_product) {
                    block_bm_orthonormalize(Q, AQ, BQ, m);
                } else {
                    block_l2_orthonormalize(Q, AQ, BQ, m);
                }
            } else {
                block_l2_orthonormalize(Q, AQ, m);
                BQ = Q;
            }
        }
        qBlocks = Q.cols() / std::max<Eigen::Index>(1, this->cfg().block_size);
    }
}

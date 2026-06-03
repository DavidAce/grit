#pragma once

#include "grit/algo/lanczos.h"
#include <algorithm>
#include <stdexcept>

namespace grit::algo {
    template<typename Scalar, grit::Form form_>
    lanczos<Scalar, form_>::lanczos(Matvec<Scalar> &A) requires(form_ == grit::Form::STANDARD)
        : Base(MatrixType{}, A) {
        this->bind_config(config);
        config.nev        = 1;
        config.block_size = 1;
        config.ncv        = std::min<Eigen::Index>(16, std::max<Eigen::Index>(1, this->N));
    }

    template<typename Scalar, grit::Form form_>
    lanczos<Scalar, form_>::lanczos(Matvec<Scalar> &A, Matvec<Scalar> &B) requires(form_ == grit::Form::GENERALIZED)
        : Base(MatrixType{}, A, B) {
        this->bind_config(config);
        config.nev        = 1;
        config.block_size = 1;
        config.ncv        = std::min<Eigen::Index>(16, std::max<Eigen::Index>(1, this->N));
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
        if(config.nev > config.block_size) throw std::runtime_error("lanczos config error: nev must not exceed block_size");
        if(config.ncv < config.nev) throw std::runtime_error("lanczos config error: ncv must be at least nev");
        if(config.ncv > this->N) throw std::runtime_error("lanczos config error: ncv must not exceed the operator size");
        if(config.block_size > config.ncv) throw std::runtime_error("lanczos config error: block_size must not exceed ncv");
        if(config.ncv % config.block_size != 0) throw std::runtime_error("lanczos config error: ncv must be divisible by block_size");
        if(config.maxRetainBlocks < 1) throw std::runtime_error("lanczos config error: maxRetainBlocks must be at least 1");
        if(config.maxRetainBlocks > config.ncv / config.block_size)
            throw std::runtime_error("lanczos config error: maxRetainBlocks must not exceed ncv / block_size");
        if(config.max_iters == 0) throw std::runtime_error("lanczos config error: max_iters must be positive or negative for unlimited");
        if(config.max_matvecs == 0) throw std::runtime_error("lanczos config error: max_matvecs must be positive or negative for unlimited");
        if(config.tol <= RealScalar{0}) throw std::runtime_error("lanczos config error: tol must be positive");
        if(config.tol_rnorm_relative < RealScalar{0}) throw std::runtime_error("lanczos config error: tol_rnorm_relative must be nonnegative");
        if(config.sat_eigval_threshold < RealScalar{0}) throw std::runtime_error("lanczos config error: sat_eigval_threshold must be nonnegative");
        if(config.sat_rnorm_threshold < RealScalar{0}) throw std::runtime_error("lanczos config error: sat_rnorm_threshold must be nonnegative");
        if(this->has_initial_guess()) {
            if(this->initial_guess().rows() != this->N) throw std::runtime_error("lanczos config error: initial guess row count must match the operator size");
            if(this->initial_guess().cols() < 1) throw std::runtime_error("lanczos config error: initial guess must have at least one column");
        }
    }

    template<typename Scalar, grit::Form form_>
    void lanczos<Scalar, form_>::run() {
        assert_config();

        this->setLogger(config.log_level, std::string("grit|") + std::string(this->form_name()));
        status.stopReason = StopReason::none;
        status.stopMessage.clear();
        status.rNorm_below_rnormTol    = false;
        status.rNorm_below_gap         = false;
        status.saturation_count_eigVal = 0;
        status.saturation_count_rNorm  = 0;
        status.rNorms_history.clear();
        status.eigVals_history.clear();
        status.matvecs_history.clear();
        status.time_orthogonalize.reset();
        status.time_orthonormalize.reset();
        status.time_orth_project.reset();
        status.time_orth_factor.reset();
        status.time_orth_update.reset();
        status.time_orth_refresh.reset();
        status.time_orth_mask.reset();
        status.time_diagonalize.reset();
        status.time_extract_ritz.reset();
        status.time_restart.reset();

        status.saturation_count_max = this->cfg().ncv;

        if(status.iter == 0) {
            status.rNorms.setOnes(this->cfg().nev);
            status.rNorms_init.setOnes(this->cfg().nev);
            status.eigVal.setOnes(this->cfg().nev);
            status.oldVal.setOnes(this->cfg().nev);
            status.absDiff.setOnes(this->cfg().nev);
            status.relDiff.setOnes(this->cfg().nev);

            Eigen::ColPivHouseholderQR<MatrixType> cpqr;
            for(long i = 0; i < 2; ++i) {
                if(V.cols() < this->cfg().block_size) {
                    auto vc = V.cols();
                    V.conservativeResize(N, this->cfg().block_size);
                    auto Vrc = V.rightCols(this->cfg().block_size - vc);
                    for(auto vj : Vrc.colwise()) { vj = Eigen::VectorXf::Random(vj.size()).template cast<Scalar>(); }
                }
                cpqr.compute(V);
                auto rank = std::min<Eigen::Index>(cpqr.rank(), this->cfg().block_size);
                V         = cpqr.householderQ().setLength(rank) * MatrixType::Identity(N, rank);
                if(V.cols() == this->cfg().block_size) break;
            }

            {
                auto m       = OrthMeta();
                m.maskPolicy = Base::MaskPolicy::COMPRESS;
                if constexpr(form_ == grit::Form::GENERALIZED) {
                    if(this->cfg().use_b_inner_product) {
                        block_bm_orthonormalize(V, AV, BV, m);
                    } else {
                        block_l2_orthonormalize(V, AV, BV, m);
                    }
                } else {
                    block_l2_orthonormalize(V, AV, m);
                    BV = V;
                }
            }
        }

        while(true) {
            this->preamble();
            build();
            this->diagonalizeT();
            extractRitzVectors();
            updateStatus();
            this->printStatus();
            run_user_callback();
            status.iter++;
            if(status.stopReason != StopReason::none) break;
        }
    }

    template<typename Scalar, grit::Form form_>
    void lanczos<Scalar, form_>::write_Q_next_B_DGKS(Eigen::Index i) {
        for(int rep = 0; rep < 2; ++rep) {
            MatrixType QjW;
            for(Eigen::Index j = i; j >= 0; --j) {
                auto Qj      = Q.middleCols(j * this->cfg().block_size, this->cfg().block_size);
                QjW          = Qj.adjoint() * W;
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
        const Eigen::Index m = this->cfg().ncv / this->cfg().block_size;

        beta_stalled = false;
        if(V.cols() != b) throw std::runtime_error("lanczos build error: V must have block_size columns");

        if(K.cols() > 0) {
            Q  = K;
            AQ = AK;
            if constexpr(form_ == grit::Form::GENERALIZED)
                BQ = BK;
            else
                BQ = Q;
        } else {
            Q  = V;
            AQ = AV;
            if constexpr(form_ == grit::Form::GENERALIZED)
                BQ = BV;
            else
                BQ = Q;
        }

        auto restart_basis = [&]() {
            auto token_restart = status.time_restart.tic_token();
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
                if(ges.info() != Eigen::Success) throw std::runtime_error("lanczos restart: generalized eigensolver failed");
                cols_ks        = std::min(cols_ks, ges.eigenvalues().size());
                auto selectIdx = this->get_ritz_indices(this->cfg().ritz, 0, cols_ks, ges.eigenvalues());

                VectorReal Y    = ges.eigenvalues()(selectIdx);
                MatrixType Z_rr = ges.eigenvectors()(Eigen::placeholders::all, selectIdx);
                MatrixType Z;
                if(this->cfg().use_refined_rayleigh_ritz) {
                    MatrixType Z_ref = get_refined_ritz_eigenvectors_gen(Z_rr, Y, AQ, BQ);
                    MatrixType Z_opt = get_optimal_rayleigh_ritz_matrix(Z_rr, Z_ref, WT1W, WT2W);
                    Z                = W * Z_opt;
                } else {
                    Z = W * Z_rr;
                }
                orthonormalize_Z(Z, T2);

                MatrixType Q_ks  = Q * Z;
                MatrixType AQ_ks = AQ * Z;
                MatrixType BQ_ks = BQ * Z;
                Q                = Q_ks;
                AQ               = AQ_ks;
                BQ               = BQ_ks;

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
                if(es.info() != Eigen::Success) throw std::runtime_error("lanczos restart: eigensolver failed");
                cols_ks        = std::clamp(std::min(config.maxRetainBlocks * this->cfg().block_size, Q.cols()), this->cfg().block_size, Q.cols());
                cols_ks        = std::min(cols_ks, es.eigenvalues().size());
                auto selectIdx = this->get_ritz_indices(this->cfg().ritz, 0, cols_ks, es.eigenvalues());

                VectorReal Y    = es.eigenvalues()(selectIdx);
                MatrixType Z_rr = es.eigenvectors()(Eigen::placeholders::all, selectIdx);
                MatrixType Z    = this->cfg().use_refined_rayleigh_ritz ? get_refined_ritz_eigenvectors_std(Z_rr, Y, Q, AQ) : Z_rr;
                orthonormalize_Z(Z, T);

                MatrixType Q_ks  = Q * Z;
                MatrixType AQ_ks = AQ * Z;
                Q                = Q_ks;
                AQ               = AQ_ks;
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

        MatrixType Q_prev;
        if(Q.cols() > b) restart_basis();
        if(Q.cols() >= 2 * b) Q_prev = Q.middleCols(Q.cols() - 2 * b, b);
        A_block.resize(b, b);
        B_block.resize(b, b);
        T.setZero(std::min(N, m * b), std::min(N, m * b));

        for(Eigen::Index i = 0; Q.cols() + b <= std::min(N, m * b); ++i) {
            auto Q_cur = Q.rightCols(b);
            W          = MultA(Q_cur);

            A_block      = Q_cur.adjoint() * W;
            W.noalias() -= Q_cur * A_block;
            if(Q_prev.cols() == b) {
                B_block      = Q_prev.adjoint() * W;
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
                auto       select_b = this->get_ritz_indices(this->cfg().ritz, 0, b, T_evals);
                VectorReal evals(b);
                for(Eigen::Index j = 0; j < b; ++j) evals(j) = T_evals(select_b[static_cast<size_t>(j)]);
                W = MultP(W, evals);
            }

            MatrixType AW = MatrixType();
            MatrixType BW = MatrixType();
            OrthMeta   meta;
            meta.maskPolicy = Base::MaskPolicy::COMPRESS;

            if constexpr(form_ == grit::Form::GENERALIZED) {
                if(config.use_b_inner_product) {
                    block_bm_orthogonalize(Q, AQ, BQ, W, AW, BW, meta, Base::RefreshMult::NO);
                    block_bm_orthonormalize(W, AW, BW, meta);
                } else {
                    block_l2_orthogonalize(Q, AQ, BQ, W, AW, BW, meta, Base::RefreshMult::NO);
                    block_l2_orthonormalize(W, AW, BW, meta);
                }
            } else {
                block_l2_orthogonalize(Q, AQ, W, AW, meta, Base::RefreshMult::NO);
                block_l2_orthonormalize(W, AW, meta);
                BW = W;
            }

            if(W.cols() < b) {
                beta_stalled = true;
                break;
            }

            Q_prev                = Q_cur;
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

        Eigen::Index k     = std::min(config.maxRetainBlocks * this->cfg().block_size, T_evals.size());
        Eigen::Index nritz = std::max({this->cfg().nev, this->cfg().block_size, k});
        status.optIdx      = this->get_ritz_indices(this->cfg().ritz, 0, nritz, T_evals);
        if(status.optIdx.empty()) {
            status.stopReason |= StopReason::subspace_exhausted;
            status.stopMessage.emplace_back("lanczos extractRitzVectors: no Ritz indices available");
            return;
        }

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
        AK     = AV.leftCols(k);
        if constexpr(form_ == grit::Form::GENERALIZED)
            BK = BV.leftCols(k);
        else
            BK = K;

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
    void lanczos<Scalar, form_>::updateStatus() {
        if(T_evals.size() < this->cfg().block_size) return;
        if(T_evals.size() < this->cfg().nev || status.optIdx.size() < static_cast<size_t>(this->cfg().nev) || status.rNorms.size() < this->cfg().nev) {
            status.stopReason |= StopReason::subspace_exhausted;
            status.stopMessage.emplace_back("lanczos updateStatus: projected problem has fewer Ritz values than requested nev");
            return;
        }
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

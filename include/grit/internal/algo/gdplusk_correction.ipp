#pragma once

#include "grit/form/base.h"
#include "grit/internal/precondition/JacobiDavidsonOperator.h"

namespace grit::algo {
    template<typename Scalar, grit::Form form_>
    std::string_view gdplusk<Scalar, form_>::ResidualCorrectionToString(ResidualCorrectionType rct) {
        switch(rct) {
            case ResidualCorrectionType::NONE: return "NONE";
            case ResidualCorrectionType::CHEAP_OLSEN: return "CHEAP_OLSEN";
            case ResidualCorrectionType::FULL_OLSEN: return "FULL_OLSEN";
            case ResidualCorrectionType::JACOBI_DAVIDSON: return "JACOBI_DAVIDSON";
            case ResidualCorrectionType::AUTO: return "AUTO";
        }
        return "NONE";
    }

    template<typename Scalar, grit::Form form_>
    typename gdplusk<Scalar, form_>::ResidualCorrectionType gdplusk<Scalar, form_>::StringToResidualCorrection(std::string_view rct) {
        if(rct == "NONE") return ResidualCorrectionType::NONE;
        if(rct == "CHEAP_OLSEN") return ResidualCorrectionType::CHEAP_OLSEN;
        if(rct == "FULL_OLSEN") return ResidualCorrectionType::FULL_OLSEN;
        if(rct == "JACOBI_DAVIDSON") return ResidualCorrectionType::JACOBI_DAVIDSON;
        if(rct == "AUTO") return ResidualCorrectionType::AUTO;
        return ResidualCorrectionType::NONE;
    }

    template<typename Scalar, grit::Form form_>
    void gdplusk<Scalar, form_>::adjust_residual_correction_type() {
        residual_correction_type_internal = config.residual_correction_type;
        if(residual_correction_type_internal == ResidualCorrectionType::AUTO) {
            if(auto_residual_correction.active == ResidualCorrectionType::JACOBI_DAVIDSON &&
               auto_residual_correction.jd_outer_iters_since_probe >= config.auto_cheap_probe_interval) {
                auto_residual_correction.iteration_method = ResidualCorrectionType::CHEAP_OLSEN;
            } else {
                auto_residual_correction.iteration_method = auto_residual_correction.active;
            }
            residual_correction_type_internal = auto_residual_correction.iteration_method;
        } else {
            auto_residual_correction.iteration_method = residual_correction_type_internal;
        }
    }

    template<typename Scalar, grit::Form form_>
    void gdplusk<Scalar, form_>::adjust_inner_tolerance([[maybe_unused]] const Eigen::Ref<const MatrixType> &S) {
        if(!config.use_adaptive_inner_tolerance) return;
        if(status.num_inner_iters_prev == 0) return;

        const auto oldtol = std::max(eps, current_inner_tol);
        const auto previous_num_inner_iters = status.num_inner_iters_prev;

        const RealScalar inner_tol_min         = eps;
        const RealScalar inner_tol_max         = RealScalar{0.75f};
        const RealScalar inner_tol_decr_factor = RealScalar{0.5f};
        const RealScalar inner_tol_incr_factor = RealScalar{2};
        const RealScalar slow_rnorm_ratio      = RealScalar{0.9f};
        const RealScalar fast_rnorm_ratio      = RealScalar{0.5f};

        const Eigen::Index low_inner_iters  = std::max<Eigen::Index>(1, config.inner_max_iters / 20);
        const Eigen::Index high_inner_iters = std::max<Eigen::Index>(low_inner_iters + 1, config.inner_max_iters / 2);

        bool       has_rnorm_progress = false;
        RealScalar rnorm_ratio        = RealScalar{1};
        if(status.rNormsAbsHistory.size() >= 2) {
            const auto &prev = status.rNormsAbsHistory[status.rNormsAbsHistory.size() - 2];
            const auto &curr = status.rNormsAbsHistory[status.rNormsAbsHistory.size() - 1];
            const auto  rows = std::min(this->cfg().nev, std::min(prev.size(), curr.size()));
            if(rows > 0) {
                VectorReal denom   = prev.topRows(rows).cwiseMax(VectorReal::Constant(rows, std::numeric_limits<RealScalar>::min()));
                VectorReal ratio   = curr.topRows(rows).cwiseQuotient(denom);
                rnorm_ratio        = ratio.maxCoeff();
                has_rnorm_progress = std::isfinite(rnorm_ratio);
            }
        }

        RealScalar next_tol = oldtol;
        if(status.outer_iter > 0) {
            if(has_rnorm_progress) {
                if(rnorm_ratio > slow_rnorm_ratio) {
                    if(previous_num_inner_iters < low_inner_iters) next_tol = oldtol * inner_tol_decr_factor;
                } else if(rnorm_ratio < fast_rnorm_ratio) {
                    if(previous_num_inner_iters > high_inner_iters) next_tol = oldtol * inner_tol_incr_factor;
                }
            } else {
                if(previous_num_inner_iters < low_inner_iters) next_tol = oldtol * inner_tol_decr_factor;
                if(previous_num_inner_iters > high_inner_iters) next_tol = oldtol * inner_tol_incr_factor;
            }
        }

        current_inner_tol = std::clamp(next_tol, inner_tol_min, inner_tol_max);
    }

    template<typename Scalar, grit::Form form_>
    void gdplusk<Scalar, form_>::update_auto_residual_correction_state() {
        if(config.residual_correction_type != ResidualCorrectionType::AUTO) return;

        auto targets = this->rNormAbsTargets();
        auto rows    = std::min({cfg().nev, status.eigVal.size(), status.rNormsAbs.size(), targets.size(), V.cols()});
        if constexpr(form_ == grit::Form::GENERALIZED) rows = std::min(rows, BV.cols());
        if(rows <= 0) return;

        VectorReal b_norms = VectorReal::Zero(rows);
        for(Eigen::Index i = 0; i < rows; ++i) {
            if constexpr(form_ == grit::Form::GENERALIZED)
                b_norms(i) = BV.col(i).norm();
            else
                b_norms(i) = V.col(i).norm();
        }

        auto outer_iteration_time = std::max(0.0, status.time_elapsed.get_time() - auto_residual_correction.outer_iteration_time_start);

        if(auto_residual_correction.active == ResidualCorrectionType::JACOBI_DAVIDSON &&
           auto_residual_correction.iteration_method == ResidualCorrectionType::CHEAP_OLSEN) {
            rows = std::min(rows, status.oldVal.size());
            VectorReal gain = VectorReal::Zero(rows);
            switch(cfg().ritz) {
                case Ritz::LR: gain = status.eigVal.topRows(rows) - status.oldVal.topRows(rows); break;
                case Ritz::SM: gain = status.oldVal.topRows(rows).cwiseAbs() - status.eigVal.topRows(rows).cwiseAbs(); break;
                case Ritz::LM: gain = status.eigVal.topRows(rows).cwiseAbs() - status.oldVal.topRows(rows).cwiseAbs(); break;
                case Ritz::NONE: [[fallthrough]];
                case Ritz::SR: gain = status.oldVal.topRows(rows) - status.eigVal.topRows(rows); break;
            }
            VectorReal residual_scales = status.rNormsAbs.topRows(rows).cwiseMax(VectorReal::Constant(rows, std::numeric_limits<RealScalar>::min()));
            VectorReal normalized_gain = gain.cwiseMax(RealScalar{0}).cwiseProduct(b_norms.topRows(rows)).cwiseQuotient(residual_scales);
            bool       keep_cheap      = false;
            for(Eigen::Index i = 0; i < rows; ++i) {
                if(status.rNormsAbs(i) > targets(i) && normalized_gain(i) > config.auto_ritz_tolerance) keep_cheap = true;
            }

            auto_residual_correction.jd_outer_iters_since_probe = 0;
            auto_residual_correction.cheap_iters                = keep_cheap ? 1 : 0;
            auto_residual_correction.active =
                keep_cheap ? ResidualCorrectionType::CHEAP_OLSEN : ResidualCorrectionType::JACOBI_DAVIDSON;
            if(keep_cheap) auto_residual_correction.jd_to_cheap_switch_outer_iters.push_back(status.outer_iter);

            if(log) {
                log->debug("auto residual correction cheap probe: {} | normalized objective gain [{}] tolerance {:.6e} | outer_iter {} mv_total {} time_iter {:.6e}s",
                           keep_cheap ? "keep CHEAP_OLSEN" : "return JACOBI_DAVIDSON",
                           fmt::join(normalized_gain.data(), normalized_gain.data() + normalized_gain.size(), ", "), config.auto_ritz_tolerance,
                           status.outer_iter, status.num_matvecs_total, outer_iteration_time);
            }
            return;
        }

        if(auto_residual_correction.iteration_method == ResidualCorrectionType::JACOBI_DAVIDSON) {
            auto_residual_correction.active      = ResidualCorrectionType::JACOBI_DAVIDSON;
            auto_residual_correction.cheap_iters = 0;
            auto_residual_correction.jd_outer_iters_since_probe++;
            return;
        }

        auto_residual_correction.active = ResidualCorrectionType::CHEAP_OLSEN;
        auto_residual_correction.cheap_iters++;
        if(status.residual_converged || auto_residual_correction.cheap_iters < static_cast<Eigen::Index>(status.max_history_size) ||
           status.eigVals_history.size() < status.max_history_size)
            return;

        VectorReal residual_scales = status.rNormsAbs.topRows(rows).cwiseMax(VectorReal::Constant(rows, std::numeric_limits<RealScalar>::min()));
        VectorReal normalized_std =
            get_standard_deviations(status.eigVals_history, false).topRows(rows).cwiseProduct(b_norms).cwiseQuotient(residual_scales);
        bool all_unconverged_localized = true;
        for(Eigen::Index i = 0; i < rows; ++i) {
            if(status.rNormsAbs(i) > targets(i) && normalized_std(i) > config.auto_ritz_tolerance) all_unconverged_localized = false;
        }

        if(log) {
            log->trace("auto residual correction Ritz localization: normalized std [{}] tolerance {:.6e} localized {} | cheap_iters {} outer_iter {} mv_total {}",
                       fmt::join(normalized_std.data(), normalized_std.data() + normalized_std.size(), ", "), config.auto_ritz_tolerance,
                       all_unconverged_localized, auto_residual_correction.cheap_iters, status.outer_iter, status.num_matvecs_total);
        }
        if(!all_unconverged_localized) return;

        auto_residual_correction.active = ResidualCorrectionType::JACOBI_DAVIDSON;
        auto_residual_correction.cheap_iters = 0;
        auto_residual_correction.jd_outer_iters_since_probe = 0;
        auto_residual_correction.cheap_to_jd_switch_outer_iters.push_back(status.outer_iter);
        if(log) {
            log->debug("auto residual correction switch: {} -> {} | reason Ritz localized | normalized std [{}] tolerance {:.6e} | outer_iter {} mv_total {} time_iter {:.6e}s",
                       ResidualCorrectionToString(ResidualCorrectionType::CHEAP_OLSEN),
                       ResidualCorrectionToString(ResidualCorrectionType::JACOBI_DAVIDSON),
                       fmt::join(normalized_std.data(), normalized_std.data() + normalized_std.size(), ", "), config.auto_ritz_tolerance,
                       status.outer_iter, status.num_matvecs_total, outer_iteration_time);
        }
    }

    template<typename Scalar, grit::Form form_>
    typename gdplusk<Scalar, form_>::MatrixType gdplusk<Scalar, form_>::cheap_Olsen_correction(const MatrixType &V, const MatrixType &S) {
        MatrixType D(S.rows(), S.cols());

        assert(V.allFinite());
        assert(S.allFinite());
        for(long i = 0; i < S.cols(); ++i) {
            auto d           = D.col(i);
            auto v           = V.col(i);
            auto s           = S.col(i);
            auto numerator   = Scalar{1};
            auto denominator = Scalar{1};

            if(this->cfg().use_b_inner_product && BV.rows() == V.rows() && BV.cols() > i) {
                auto bv     = BV.col(i);
                numerator   = bv.dot(s);
                denominator = bv.dot(v);
            } else {
                numerator   = v.dot(s);
                denominator = v.dot(v);
            }

            auto delta  = std::abs(denominator) > eps * 100 ? numerator / denominator : RealScalar{0};
            d.noalias() = s - delta * v;
        }
        return D;
    }

    template<typename Scalar, grit::Form form_>
    typename gdplusk<Scalar, form_>::MatrixType gdplusk<Scalar, form_>::full_Olsen_correction(const MatrixType &V, const MatrixType &S) {
        MatrixType MV;
        MatrixType MS;
        MatrixType coeffs;
        auto       Y = T_evals(status.optIdx);

        if(this->cfg().use_b_inner_product && BV.rows() == V.rows() && BV.cols() == V.cols()) {
            MV.noalias() = A.has_preconditioner_apply() ? MultP(V, Y) : V;
            MS.noalias() = A.has_preconditioner_apply() ? MultP(S, Y) : S;

            MatrixType G       = BV.adjoint() * MV;
            MatrixType VT_B_MS = BV.adjoint() * MS;
            coeffs             = G.ldlt().solve(VT_B_MS);
        } else {
            MV.noalias() = A.has_preconditioner_apply() ? MultP(V, Y) : V;
            MS.noalias() = A.has_preconditioner_apply() ? MultP(S, Y) : S;

            MatrixType G     = V.adjoint() * MV;
            MatrixType VT_MS = V.adjoint() * MS;
            coeffs           = G.ldlt().solve(VT_MS);
        }
        return MS - MV * coeffs;
    }

    template<typename Scalar, grit::Form form_>
    typename gdplusk<Scalar, form_>::MatrixType gdplusk<Scalar, form_>::jacobi_davidson_l2_correction(const MatrixType &V, const MatrixType &S,
                                                                                                      const VectorReal &evals) {
        assert(V.rows() == S.rows());
        assert(V.cols() == S.cols());
        assert(!this->cfg().use_b_inner_product);

        auto MatrixOp = [this](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
            if constexpr(form_ == grit::Form::GENERALIZED)
                return MultB_inner(X);
            else
                return MultA_inner(X);
        };

        auto ProjectOpL = [this, &V](const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> Y) -> void {
            auto                    t_project_left = status.time_project_left_inner.tic_token();
            thread_local MatrixType T;
            T.resize(V.cols(), X.cols());
            Y.resize(X.rows(), X.cols());
            T.noalias() = V.adjoint() * X;
            Y.noalias() = X - V * T;
        };
        auto ProjectOpL_tmp = [ProjectOpL](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
            MatrixType Y(X.rows(), X.cols());
            ProjectOpL(X, Y);
            return Y;
        };

        auto ProjectOpR = [this, &V](const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> Y) -> void {
            auto                    t_project_right = status.time_project_right_inner.tic_token();
            thread_local MatrixType T;
            T.resize(V.cols(), X.cols());
            Y.resize(X.rows(), X.cols());
            T.noalias() = V.adjoint() * X;
            Y.noalias() = X - V * T;
        };
        auto ProjectOpR_tmp = [ProjectOpR](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
            MatrixType Y(X.rows(), X.cols());
            ProjectOpR(X, Y);
            return Y;
        };
        MatrixType RHS = -ProjectOpL_tmp(S);

        if(D.size() != RHS.size()) D.setZero(RHS.rows(), RHS.cols());

        for(Eigen::Index i = 0; i < S.cols(); ++i) {
            auto              d   = D.col(i);
            RealScalar        th  = evals(i);
            const VectorType &rhs = RHS.col(i);

            if(i > 0) {
                const VectorType &s  = S.col(i);
                const VectorType &v  = V.col(i);
                auto              ev = evals.middleRows(i, 1);
                D.col(i).noalias()   = MultP(s, ev);
                D.col(i).noalias()   = cheap_Olsen_correction(v, D.col(i));
            } else {
                if(A.has_preconditioner_update()) {
                    A.preconditioner_update(th);
                    status.time_preconditioner_update_inner += A.t_precond_update->get_last_interval();
                    status.num_preconditioner_updates_inner++;
                }

                IterativeLinearSolverConfig<Scalar> cfg;
                cfg.result               = {};
                cfg.matdef               = MatDef::IND;
                cfg.max_inner_iters      = config.inner_max_iters;
                cfg.tolerance            = current_inner_tol;
                cfg.theta                = th;
                cfg.preconditioner_apply = [this](const Eigen::Ref<const VectorType> &x, Eigen::Ref<VectorType> y, RealScalar theta) -> void {
                    A.preconditioner_apply(x, y, theta);
                    if(A.has_preconditioner_apply()) {
                        status.time_preconditioner_apply_inner += A.t_precond->get_last_interval();
                        status.num_preconditioner_apply_inner++;
                    }
                };

                auto ResidualOp = [this, th](const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> HX) -> void {
                    HX.resize(X.rows(), X.cols());
                    if constexpr(form_ == grit::Form::GENERALIZED) {
                        if(config.use_jd_b_only) {
                            HX.noalias() = MultB_inner(X);
                        } else {
                            HX.noalias() = MultA_inner(X) - th * MultB_inner(X);
                        }
                    } else {
                        HX.noalias() = MultA_inner(X) - th * X;
                    }
                };

                auto JDop = internal::precondition::JacobiDavidsonOperator<Scalar>(rhs.rows(), ResidualOp, ProjectOpL, ProjectOpR, MatrixOp);

                d.noalias() = internal::precondition::JacobiDavidsonSolver(JDop, rhs, cfg);
                d.noalias() = ProjectOpR_tmp(d);

                status.num_inner_iters          += cfg.result.num_inner_iters;
                status.num_operator_inner       += cfg.result.matvecs;
                status.num_precond_inner        += cfg.result.precond;
                status.time_solve_inner          += cfg.result.time;
                status.time_operator_inner       += cfg.result.time_matvecs;
                status.time_preconditioner_inner += cfg.result.time_precond;
                status.inner_error_last          = std::max(status.inner_error_last, cfg.result.error);
                status.inner_tol_last            = std::max(status.inner_tol_last, cfg.tolerance);
            }
        }
        return D;
    }

    template<typename Scalar, grit::Form form_>
    typename gdplusk<Scalar, form_>::MatrixType gdplusk<Scalar, form_>::jacobi_davidson_bm_correction(const MatrixType &V, const MatrixType &BV,
                                                                                                      const MatrixType &S, const VectorReal &evals)
        requires(form_ == grit::Form::GENERALIZED)
    {
        assert(this->cfg().use_b_inner_product);
        assert(V.rows() == S.rows());
        assert(V.cols() == S.cols());
        assert(BV.size() == V.size());

        auto ProjectOpL = [this, &V, &BV](const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> Y) -> void {
            auto                    t_project_left = status.time_project_left_inner.tic_token();
            thread_local MatrixType T;
            T.resize(V.cols(), X.cols());
            Y.resize(X.rows(), X.cols());
            T.noalias()  = V.adjoint() * X;
            Y.noalias()  = X;
            Y.noalias() -= BV * T;
        };
        auto ProjectOpL_tmp = [ProjectOpL](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
            MatrixType Y(X.rows(), X.cols());
            ProjectOpL(X, Y);
            return Y;
        };

        auto ProjectOpR = [this, &V, &BV](const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> Y) -> void {
            auto                    t_project_right = status.time_project_right_inner.tic_token();
            thread_local MatrixType T;
            T.resize(BV.cols(), X.cols());
            Y.resize(X.rows(), X.cols());
            T.noalias()  = BV.adjoint() * X;
            Y.noalias()  = X;
            Y.noalias() -= V * T;
        };
        auto ProjectOpR_tmp = [ProjectOpR](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
            MatrixType Y(X.rows(), X.cols());
            ProjectOpR(X, Y);
            return Y;
        };
        MatrixType RHS = -ProjectOpL_tmp(S);

        MatrixType D(S.rows(), S.cols());

        for(Eigen::Index i = 0; i < S.cols(); ++i) {
            auto              d   = D.col(i);
            const RealScalar &th  = evals(i);
            const VectorType &rhs = RHS.col(i);
            if(i >= cfg().nev) {
                const VectorType &s  = S.col(i);
                const VectorType &v  = V.col(i);
                auto              ev = evals.middleRows(i, 1);
                D.col(i).noalias()   = MultP(s, ev);
                D.col(i).noalias()   = cheap_Olsen_correction(v, D.col(i));
            } else {
                if(A.has_preconditioner_update()) {
                    A.preconditioner_update(th);
                    status.time_preconditioner_update_inner += A.t_precond_update->get_last_interval();
                    status.num_preconditioner_updates_inner++;
                }

                IterativeLinearSolverConfig<Scalar> cfg;
                cfg.result               = {};
                cfg.matdef               = MatDef::IND;
                cfg.max_inner_iters      = config.inner_max_iters;
                cfg.tolerance            = current_inner_tol;
                cfg.theta                = th;
                cfg.preconditioner_apply = [this](const Eigen::Ref<const VectorType> &x, Eigen::Ref<VectorType> y, RealScalar theta) -> void {
                    A.preconditioner_apply(x, y, theta);
                    if(A.has_preconditioner_apply()) {
                        status.time_preconditioner_apply_inner += A.t_precond->get_last_interval();
                        status.num_preconditioner_apply_inner++;
                    }
                };

                auto MatrixOp = [this](const Eigen::Ref<const MatrixType> &X) -> MatrixType { return MultB_inner(X); };

                auto ResidualOp = [this, th](const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> HX) -> void {
                    HX.resize(X.rows(), X.cols());
                    if(config.use_jd_b_only) {
                        HX.noalias() = MultB_inner(X);
                    } else {
                        HX.noalias() = MultA_inner(X) - th * MultB_inner(X);
                    }
                };

                auto JDop = internal::precondition::JacobiDavidsonOperator<Scalar>(rhs.rows(), ResidualOp, ProjectOpL, ProjectOpR, MatrixOp);

                d.noalias() = internal::precondition::JacobiDavidsonSolver(JDop, rhs, cfg);
                d.noalias() = ProjectOpR_tmp(d);

                status.num_inner_iters          += cfg.result.num_inner_iters;
                status.num_operator_inner       += cfg.result.matvecs;
                status.num_precond_inner        += cfg.result.precond;
                status.time_solve_inner          += cfg.result.time;
                status.time_operator_inner       += cfg.result.time_matvecs;
                status.time_preconditioner_inner += cfg.result.time_precond;
                status.inner_error_last          = std::max(status.inner_error_last, cfg.result.error);
                status.inner_tol_last            = std::max(status.inner_tol_last, cfg.tolerance);
            }
        }
        return D;
    }

    template<typename Scalar, grit::Form form_>
    typename gdplusk<Scalar, form_>::MatrixType gdplusk<Scalar, form_>::get_sBlock(const MatrixType &S_in) {
        auto       t_residual_correction = status.time_residual_correction.tic_token();
        MatrixType S = S_in;
        assert(S.allFinite());
        assert(S.cols() > 0);
        auto Y = T_evals(status.optIdx);

        switch(residual_correction_type_internal) {
            case ResidualCorrectionType::NONE:
                if(A.has_preconditioner_apply()) { S = MultP(S, Y); }
                break;
            case ResidualCorrectionType::AUTO: [[fallthrough]];
            case ResidualCorrectionType::CHEAP_OLSEN:
                if(A.has_preconditioner_apply()) { S = MultP(S, Y); }
                S.noalias() = cheap_Olsen_correction(V, S);
                break;
            case ResidualCorrectionType::FULL_OLSEN: S.noalias() = full_Olsen_correction(V, S); break;
            case ResidualCorrectionType::JACOBI_DAVIDSON:
                if constexpr(form_ == grit::Form::GENERALIZED) {
                    if(this->cfg().use_b_inner_product) {
                        S.noalias() = jacobi_davidson_bm_correction(V, BV, S, Y);
                    } else {
                        S.noalias() = jacobi_davidson_l2_correction(V, S, Y);
                    }
                } else {
                    S.noalias() = jacobi_davidson_l2_correction(V, S, Y);
                }
                break;
        }
        assert_allFinite(S);
        return S;
    }

}

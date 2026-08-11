#pragma once

#include "grit/form/base.h"
#include "grit/internal/precondition/JacobiDavidsonOperator.h"

namespace grit::algo {
    template<typename Scalar, grit::Form form_> void gdplusk<Scalar, form_>::adjust_residual_correction_type() {
        residual_correction_type_internal = config.residual_correction_type;
        if(residual_correction_type_internal == ResidualCorrectionType::AUTO) {
            auto         samples    = this->get_num_consecutive_correction_samples(ResidualCorrectionType::JACOBI_DAVIDSON);
            auto         slopes     = get_rnorm_slopes(samples);
            auto         targets    = this->rNormAbsTargets();
            auto         rows       = std::min({cfg().nev, slopes.size(), status.rNormsAbs.size(), targets.size()});
            Eigen::Index probe_slot = -1;
            for(Eigen::Index i = 0; i < rows; ++i) {
                if(status.rNormsAbs(i) * current_inner_tol > targets(i) && std::isfinite(slopes(i)) && slopes(i) >= RealScalar{0}) {
                    probe_slot = i;
                    break;
                }
            }
            bool probe_in_progress =
                auto_residual_correction.active == ResidualCorrectionType::JACOBI_DAVIDSON && auto_residual_correction.cheap_olsen_iters > 0;
            bool probe_available = config.auto_max_probes < 0 || auto_residual_correction.probes_started < config.auto_max_probes;
            if(auto_residual_correction.active == ResidualCorrectionType::JACOBI_DAVIDSON && (probe_in_progress || (probe_available && probe_slot >= 0))) {
                auto_residual_correction.iteration_method = ResidualCorrectionType::CHEAP_OLSEN;
            } else {
                auto_residual_correction.iteration_method = auto_residual_correction.active;
            }
            if(log && auto_residual_correction.active == ResidualCorrectionType::JACOBI_DAVIDSON && !probe_in_progress && probe_available && probe_slot >= 0) {
                log->debug("AUTO probe start: JD | log10|r| slope={::.3e} samples={} |rNorm|={::.3e} tgt={::.3e} slot={} | it={} mv={}", slopes, samples,
                           status.rNormsAbs, targets, probe_slot, status.outer_iter, status.num_matvecs_total);
            }
            residual_correction_type_internal = auto_residual_correction.iteration_method;
        } else {
            auto_residual_correction.iteration_method = residual_correction_type_internal;
        }
    }

    template<typename Scalar, grit::Form form_> void gdplusk<Scalar, form_>::adjust_inner_tolerance([[maybe_unused]] const Eigen::Ref<const MatrixType> &S) {
        if(!config.use_adaptive_inner_tolerance) return;
        if(status.num_inner_iters_prev == 0) return;

        const auto oldtol                   = std::max(eps, current_inner_tol);
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
        if(status.history.size() >= 2) {
            const auto &prev = status.history[status.history.size() - 2].rnorms;
            const auto &curr = status.history.back().rnorms;
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

    template<typename Scalar, grit::Form form_> void gdplusk<Scalar, form_>::update_auto_residual_correction_state() {
        if(config.residual_correction_type != ResidualCorrectionType::AUTO) return;

        auto rows = std::min({cfg().nev, status.eigVal.size(), status.rNormsAbs.size()});
        if(rows <= 0) return;

        auto         ritz_samples          = this->get_num_consecutive_correction_samples(auto_residual_correction.iteration_method);
        const bool   ritz_progress_ready   = ritz_samples >= Base::min_saturation_samples;
        VectorReal   ritz_slopes           = VectorReal::Constant(rows, std::numeric_limits<RealScalar>::quiet_NaN());
        VectorReal   drift_thresholds      = VectorReal::Constant(rows, std::numeric_limits<RealScalar>::quiet_NaN());
        Eigen::Index ritz_suffix_samples   = 0;
        bool         ritz_progress_stopped = false;
        if(ritz_progress_ready) {
            drift_thresholds      = this->get_ritz_drift_thresholds(rows);
            ritz_progress_stopped = this->eigVals_have_saturated(&ritz_slopes, &ritz_suffix_samples, nullptr, ritz_samples);
        }
        const auto saturation_count_switch = std::max<Eigen::Index>(1, status.saturation_count_max / 2);
        const bool ritz_switch_ready       = ritz_progress_stopped && status.saturation_count_eigVal >= saturation_count_switch;

        auto        outer_iteration_time = std::max(0.0, status.time_elapsed.get_time() - auto_residual_correction.outer_iteration_time_start);
        const auto  num_matvecs_iter     = status.num_matvecs + status.num_matvecs_inner;
        const auto  num_precond_iter     = status.num_precond + status.num_precond_inner;
        std::string pcMsg;
        if(num_precond_iter > 0) pcMsg = fmt::format(" pc={}|{}", num_precond_iter, status.num_precond_total);

        // The active method identifies the current phase; iteration_method records the correction just used. Cheap Olsen iterations
        // selected during the JD phase form a probe. Complete its requested length, then use the same Ritz-drift test to continue
        // Cheap Olsen or return to JD.
        if(auto_residual_correction.active == ResidualCorrectionType::JACOBI_DAVIDSON &&
           auto_residual_correction.iteration_method == ResidualCorrectionType::CHEAP_OLSEN) {
            if(auto_residual_correction.cheap_olsen_iters == 0) { auto_residual_correction.probes_started++; }
            auto_residual_correction.cheap_olsen_iters++;
            if(auto_residual_correction.cheap_olsen_iters < config.auto_probe_length || (ritz_progress_stopped && !ritz_switch_ready)) {
                if(log) {
                    log->trace("AUTO probe: {}/{} | Ritz sat={}/{} stop={} suffix={} slope/mv {::.3e} threshold/mv={::.3e} | it={} mv={}",
                               auto_residual_correction.cheap_olsen_iters, config.auto_probe_length, status.saturation_count_eigVal, saturation_count_switch,
                               status.saturation_count_max, ritz_suffix_samples, ritz_slopes, drift_thresholds, status.outer_iter, status.num_matvecs_total);
                }
                return;
            }

            auto probe_iters      = auto_residual_correction.cheap_olsen_iters;
            bool keep_cheap_olsen = !ritz_progress_stopped;
            if(!keep_cheap_olsen) auto_residual_correction.cheap_olsen_iters = 0;
            auto_residual_correction.active = keep_cheap_olsen ? ResidualCorrectionType::CHEAP_OLSEN : ResidualCorrectionType::JACOBI_DAVIDSON;
            if(keep_cheap_olsen) {
                auto_residual_correction.probes_started = 0;
                auto_residual_correction.jd_to_cheap_olsen_switch_outer_iters.push_back(status.outer_iter);
            }

            if(log) {
                log->debug("AUTO probe: {} after {} iterations | Ritz slope/mv {::.3e} threshold/mv={::.3e} sat={}/{} stop={} suffix={} tol={:.3e} | "
                           "it={} mv={}|{}{} t={:.1e}|{:.1e}s",
                           keep_cheap_olsen ? "keep CHEAP_OLSEN" : "return JACOBI_DAVIDSON", probe_iters, ritz_slopes, drift_thresholds,
                           status.saturation_count_eigVal, saturation_count_switch, status.saturation_count_max, ritz_suffix_samples,
                           config.ritz_saturation_tolerance, status.outer_iter, num_matvecs_iter, status.num_matvecs_total, pcMsg, outer_iteration_time,
                           status.time_elapsed.get_time());
            }
            return;
        }

        // Keep JD active; the history tags determine how many consecutive JD samples are available for the next probe decision.
        if(auto_residual_correction.iteration_method == ResidualCorrectionType::JACOBI_DAVIDSON) {
            auto_residual_correction.active            = ResidualCorrectionType::JACOBI_DAVIDSON;
            auto_residual_correction.cheap_olsen_iters = 0;
            return;
        }

        // During the initial or resumed Cheap Olsen phase, wait for a usable Ritz history and hand control to JD once no
        // unconverged Ritz value has directed progress.
        auto_residual_correction.active = ResidualCorrectionType::CHEAP_OLSEN;
        auto_residual_correction.cheap_olsen_iters++;
        if(status.residual_converged || !ritz_progress_ready) return;

        if(log) {
            log->trace("AUTO Cheap Olsen: Ritz slope/mv {::.3e} threshold/mv={::.3e} sat={}/{} stop={} suffix={} tol={:.3e} | CO iters={} it={} mv={}",
                       ritz_slopes, drift_thresholds, status.saturation_count_eigVal, saturation_count_switch, status.saturation_count_max, ritz_suffix_samples,
                       config.ritz_saturation_tolerance, auto_residual_correction.cheap_olsen_iters, status.outer_iter, status.num_matvecs_total);
        }
        if(!ritz_switch_ready) return;

        auto_residual_correction.active            = ResidualCorrectionType::JACOBI_DAVIDSON;
        auto_residual_correction.cheap_olsen_iters = 0;
        auto_residual_correction.cheap_olsen_to_jd_switch_outer_iters.push_back(status.outer_iter);
        if(log) {
            log->info("AUTO switch: {} -> {} | Ritz slope/mv {::.3e} threshold/mv={::.3e} sat={}/{} stop={} suffix={} tol={:.3e} | "
                      "it={} mv={}|{}{} t={:.1e}|{:.1e}s",
                      enum2sv(ResidualCorrectionType::CHEAP_OLSEN), enum2sv(ResidualCorrectionType::JACOBI_DAVIDSON), ritz_slopes, drift_thresholds,
                      status.saturation_count_eigVal, saturation_count_switch, status.saturation_count_max, ritz_suffix_samples,
                      config.ritz_saturation_tolerance, status.outer_iter, num_matvecs_iter, status.num_matvecs_total, pcMsg, outer_iteration_time,
                      status.time_elapsed.get_time());
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

                auto JDop = internal::precondition::JacobiDavidsonOperator<Scalar>(rhs.rows(), ResidualOp, ProjectOpL, ProjectOpR);

                d.noalias() = internal::precondition::JacobiDavidsonSolver(JDop, rhs, cfg);
                d.noalias() = ProjectOpR_tmp(d);

                status.num_inner_iters     += cfg.result.num_inner_iters;
                status.num_operator_inner  += cfg.result.matvecs;
                status.time_solve_inner    += cfg.result.time;
                status.time_operator_inner += cfg.result.time_matvecs;
                if(A.has_preconditioner_apply()) {
                    status.num_precond_inner         += cfg.result.precond;
                    status.time_preconditioner_inner += cfg.result.time_precond;
                }
                status.inner_error_last = std::max(status.inner_error_last, cfg.result.error);
                status.inner_tol_last   = std::max(status.inner_tol_last, cfg.tolerance);
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

                auto ResidualOp = [this, th](const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> HX) -> void {
                    HX.resize(X.rows(), X.cols());
                    if(config.use_jd_b_only) {
                        HX.noalias() = MultB_inner(X);
                    } else {
                        HX.noalias() = MultA_inner(X) - th * MultB_inner(X);
                    }
                };

                auto JDop = internal::precondition::JacobiDavidsonOperator<Scalar>(rhs.rows(), ResidualOp, ProjectOpL, ProjectOpR);

                d.noalias() = internal::precondition::JacobiDavidsonSolver(JDop, rhs, cfg);
                d.noalias() = ProjectOpR_tmp(d);

                status.num_inner_iters     += cfg.result.num_inner_iters;
                status.num_operator_inner  += cfg.result.matvecs;
                status.time_solve_inner    += cfg.result.time;
                status.time_operator_inner += cfg.result.time_matvecs;
                if(A.has_preconditioner_apply()) {
                    status.num_precond_inner         += cfg.result.precond;
                    status.time_preconditioner_inner += cfg.result.time_precond;
                }
                status.inner_error_last = std::max(status.inner_error_last, cfg.result.error);
                status.inner_tol_last   = std::max(status.inner_tol_last, cfg.tolerance);
            }
        }
        return D;
    }

    template<typename Scalar, grit::Form form_> typename gdplusk<Scalar, form_>::MatrixType gdplusk<Scalar, form_>::get_sBlock(const MatrixType &S_in) {
        auto       t_residual_correction = status.time_residual_correction.tic_token();
        MatrixType S                     = S_in;
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

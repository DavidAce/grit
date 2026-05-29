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
               auto_residual_correction.jd_steps_since_probe >= config.auto_cheap_probe_interval) {
                auto_residual_correction.step_method = ResidualCorrectionType::CHEAP_OLSEN;
            } else {
                auto_residual_correction.step_method = auto_residual_correction.active;
            }
            residual_correction_type_internal = auto_residual_correction.step_method;
        } else {
            auto_residual_correction.step_method = residual_correction_type_internal;
        }
    }

    template<typename Scalar, grit::Form form_>
    void gdplusk<Scalar, form_>::adjust_inner_tolerance([[maybe_unused]] const Eigen::Ref<const MatrixType> &S) {
        if(!config.use_adaptive_inner_tolerance) return;
        if(status.num_iters_inner_prev == 0) return;

        const auto oldtol = std::max(eps, current_inner_tol);
        const auto oldits = status.num_iters_inner_prev;

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
        if(status.rNorms_history.size() >= 2) {
            const auto &prev = status.rNorms_history[status.rNorms_history.size() - 2];
            const auto &curr = status.rNorms_history[status.rNorms_history.size() - 1];
            const auto  rows = std::min(this->cfg().nev, std::min(prev.size(), curr.size()));
            if(rows > 0) {
                VectorReal denom   = prev.topRows(rows).cwiseMax(VectorReal::Constant(rows, std::numeric_limits<RealScalar>::min()));
                VectorReal ratio   = curr.topRows(rows).cwiseQuotient(denom);
                rnorm_ratio        = ratio.maxCoeff();
                has_rnorm_progress = std::isfinite(rnorm_ratio);
            }
        }

        RealScalar next_tol = oldtol;
        if(status.iter > 0) {
            if(has_rnorm_progress) {
                if(rnorm_ratio > slow_rnorm_ratio) {
                    if(oldits < low_inner_iters) next_tol = oldtol * inner_tol_decr_factor;
                } else if(rnorm_ratio < fast_rnorm_ratio) {
                    if(oldits > high_inner_iters) next_tol = oldtol * inner_tol_incr_factor;
                }
            } else {
                if(oldits < low_inner_iters) next_tol = oldtol * inner_tol_decr_factor;
                if(oldits > high_inner_iters) next_tol = oldtol * inner_tol_incr_factor;
            }
        }

        current_inner_tol = std::clamp(next_tol, inner_tol_min, inner_tol_max);
    }

    template<typename Scalar, grit::Form form_>
    typename gdplusk<Scalar, form_>::RealScalar gdplusk<Scalar, form_>::get_auto_rnorm_scalar(const VectorReal &rnorms) const {
        if(rnorms.size() == 0) return RealScalar{0};
        auto rows = std::min<Eigen::Index>(cfg().nev, rnorms.size());
        return rnorms.topRows(rows).maxCoeff();
    }

    template<typename Scalar, grit::Form form_>
    typename gdplusk<Scalar, form_>::RealScalar gdplusk<Scalar, form_>::get_auto_probe_eigval_improvement() const {
        if(status.oldVal.size() == 0 || status.eigVal.size() == 0) return RealScalar{0};
        auto before = status.oldVal(0);
        auto after  = status.eigVal(0);
        switch(cfg().ritz) {
            case OptRitz::LR: return after - before;
            case OptRitz::SM: return std::abs(before) - std::abs(after);
            case OptRitz::LM: return std::abs(after) - std::abs(before);
            case OptRitz::NONE: [[fallthrough]];
            case OptRitz::SR: return before - after;
        }
        return RealScalar{0};
    }

    template<typename Scalar, grit::Form form_>
    typename gdplusk<Scalar, form_>::AutoSaturationInfo gdplusk<Scalar, form_>::get_auto_rnorm_saturation_info() {
        AutoSaturationInfo info;
        info.enabled      = config.auto_sat_rnorm_threshold > RealScalar{0};
        info.threshold    = config.auto_sat_rnorm_threshold;
        info.history_size = static_cast<Eigen::Index>(status.rNorms_history.size());
        if(!info.enabled) return info;

        const auto min_history_size = std::min<size_t>(status.max_history_size, size_t{2});
        info.enough_history         = status.iter >= static_cast<Eigen::Index>(min_history_size) && status.rNorms_history.size() >= min_history_size;
        if(!info.enough_history || status.rNorms.size() == 0) return info;

        auto rows = std::min<Eigen::Index>(cfg().nev, status.rNorms.size());
        if(rows <= 0) return info;
        VectorReal             scales = relative_rNormScales().topRows(rows);
        std::deque<VectorReal> relative_history;
        for(const auto &history : status.rNorms_history) {
            if(history.size() < rows) throw std::runtime_error("rNorms_history has unequal size vectors");
            relative_history.emplace_back(history.topRows(rows).cwiseQuotient(scales));
        }
        VectorReal stds  = get_standard_deviations(relative_history, false);
        VectorReal vals  = status.rNorms.topRows(rows).cwiseQuotient(scales);
        VectorReal scale = vals.cwiseMax(VectorReal::Constant(rows, std::numeric_limits<RealScalar>::min()));
        VectorReal ratio = stds.topRows(rows).cwiseQuotient(scale);

        info.value     = vals.maxCoeff();
        info.stddev    = stds.topRows(rows).maxCoeff();
        info.scale     = scale.maxCoeff();
        info.ratio     = ratio.maxCoeff();
        info.saturated = (ratio.array() < info.threshold).all();
        return info;
    }

    template<typename Scalar, grit::Form form_>
    typename gdplusk<Scalar, form_>::AutoSaturationInfo gdplusk<Scalar, form_>::get_auto_eigval_saturation_info() {
        AutoSaturationInfo info;
        info.enabled      = config.auto_sat_eigval_threshold > RealScalar{0};
        info.threshold    = config.auto_sat_eigval_threshold;
        info.history_size = static_cast<Eigen::Index>(status.eigVals_history.size());
        if(!info.enabled) return info;

        const auto min_history_size = std::min<size_t>(status.max_history_size, size_t{2});
        info.enough_history         = status.iter >= static_cast<Eigen::Index>(min_history_size) && status.eigVals_history.size() >= min_history_size;
        if(!info.enough_history || status.eigVal.size() == 0) return info;

        VectorReal stds = get_standard_deviations(status.eigVals_history, false);
        auto       rows = std::min({cfg().nev, status.eigVal.size(), stds.size()});
        if(rows <= 0) return info;
        VectorReal vals  = status.eigVal.topRows(rows).cwiseAbs();
        VectorReal scale = VectorReal::Zero(rows);
        for(const auto &history : status.eigVals_history) {
            if(history.size() < rows) throw std::runtime_error("eigVals_history has unequal size vectors");
            scale.array() += history.topRows(rows).array().abs();
        }
        scale            /= static_cast<RealScalar>(status.eigVals_history.size());
        scale             = scale.cwiseMax(VectorReal::Constant(rows, std::numeric_limits<RealScalar>::min()));
        VectorReal ratio  = stds.topRows(rows).cwiseQuotient(scale);

        info.value     = vals.maxCoeff();
        info.stddev    = stds.topRows(rows).maxCoeff();
        info.scale     = scale.maxCoeff();
        info.ratio     = ratio.maxCoeff();
        info.saturated = (ratio.array() < info.threshold).all();
        return info;
    }

    template<typename Scalar, grit::Form form_>
    typename gdplusk<Scalar, form_>::AutoSaturationStatus gdplusk<Scalar, form_>::get_auto_saturation_status() {
        AutoSaturationStatus status_auto;
        status_auto.eigval = get_auto_eigval_saturation_info();
        status_auto.rnorm  = get_auto_rnorm_saturation_info();
        status_auto.ready  = status_auto.eigval.saturated && status_auto.rnorm.saturated;
        return status_auto;
    }

    template<typename Scalar, grit::Form form_>
    bool gdplusk<Scalar, form_>::auto_jd_start_ready() {
        return get_auto_saturation_status().ready || (config.auto_jd_start_rnorm_threshold > RealScalar{0} && status.rNorms.size() > 0 &&
                                                      get_auto_rnorm_scalar(relative_rNorms(status.rNorms)) <= config.auto_jd_start_rnorm_threshold);
    }

    template<typename Scalar, grit::Form form_>
    void gdplusk<Scalar, form_>::update_auto_residual_correction_state() {
        if(config.residual_correction_type != ResidualCorrectionType::AUTO) return;

        auto method_name = [](ResidualCorrectionType method) -> std::string_view {
            switch(method) {
                case ResidualCorrectionType::CHEAP_OLSEN: return "CHEAP_OLSEN";
                case ResidualCorrectionType::JACOBI_DAVIDSON: return "JACOBI_DAVIDSON";
                case ResidualCorrectionType::NONE: return "NONE";
                case ResidualCorrectionType::FULL_OLSEN: return "FULL_OLSEN";
                case ResidualCorrectionType::AUTO: return "AUTO";
            }
            return "NONE";
        };

        auto step_seconds = std::max(0.0, status.time_elapsed.get_time() - auto_residual_correction.step_time_start);

        if(auto_residual_correction.active == ResidualCorrectionType::JACOBI_DAVIDSON &&
           auto_residual_correction.step_method == ResidualCorrectionType::CHEAP_OLSEN) {
            auto rnorm = get_auto_rnorm_scalar(status.rNorms);
            // Cheap probes are accepted only when they move the Ritz value by more than
            // the variance scale implied by the stored absolute residual norm.
            auto rnorm_squared     = rnorm * rnorm;
            auto op_norm_estimate  = get_op_norm_estimate(status.eigVal.size() > 0 ? std::optional<RealScalar>{status.eigVal(0)} : std::nullopt);
            auto probe_scale_floor = eps * op_norm_estimate;
            auto probe_scale       = std::max(rnorm_squared, probe_scale_floor);
            auto improvement       = get_auto_probe_eigval_improvement();
            auto threshold         = config.auto_cheap_probe_factor * probe_scale;
            bool keep_cheap        = improvement > threshold;

            if(keep_cheap) {
                auto_residual_correction.active               = ResidualCorrectionType::CHEAP_OLSEN;
                auto_residual_correction.dwell                = 0;
                auto_residual_correction.jd_steps_since_probe = 0;
                auto_residual_correction.jd_to_cheap_switch_iters.push_back(status.iter);
            } else {
                auto_residual_correction.active               = ResidualCorrectionType::JACOBI_DAVIDSON;
                auto_residual_correction.jd_steps_since_probe = 0;
            }

            if(eiglog) {
                eiglog->debug("auto residual correction cheap probe: {} -> {} | eigval {:.16e}->{:.16e} improvement {:.6e} threshold {:.6e} factor {:.6e} "
                              "abs rnorm {:.6e} abs rnorm2 {:.6e} scale_floor {:.6e} interval {} decision {} | mv {} outer {} inner {} inner_iters {} jdops {} "
                              "time {:.6e}s",
                              method_name(ResidualCorrectionType::JACOBI_DAVIDSON), method_name(ResidualCorrectionType::CHEAP_OLSEN),
                              status.oldVal.size() > 0 ? status.oldVal(0) : RealScalar{0}, status.eigVal.size() > 0 ? status.eigVal(0) : RealScalar{0},
                              improvement, threshold, config.auto_cheap_probe_factor, rnorm, rnorm_squared, probe_scale_floor, config.auto_cheap_probe_interval,
                              keep_cheap ? "keep CHEAP_OLSEN" : "return JACOBI_DAVIDSON", status.num_matvecs + status.num_matvecs_inner, status.num_matvecs,
                              status.num_matvecs_inner, status.num_iters_inner, status.num_jdops_inner, step_seconds);
            }
            return;
        }

        if(auto_residual_correction.step_method == ResidualCorrectionType::JACOBI_DAVIDSON) {
            auto_residual_correction.active = ResidualCorrectionType::JACOBI_DAVIDSON;
            auto_residual_correction.jd_steps_since_probe++;
            if(eiglog) {
                eiglog->trace("auto residual correction jd step: steps since cheap probe {}/{}", auto_residual_correction.jd_steps_since_probe,
                              config.auto_cheap_probe_interval);
            }
            return;
        }

        auto_residual_correction.active = ResidualCorrectionType::CHEAP_OLSEN;
        auto_residual_correction.dwell++;
        auto saturation             = get_auto_saturation_status();
        auto rrnorm                 = get_auto_rnorm_scalar(relative_rNorms(status.rNorms));
        bool jd_start_rnorm_enabled = config.auto_jd_start_rnorm_threshold > RealScalar{0};
        bool jd_start_rnorm_ready   = jd_start_rnorm_enabled && status.rNorms.size() > 0 && rrnorm <= config.auto_jd_start_rnorm_threshold;
        bool jd_start_ready         = saturation.ready || jd_start_rnorm_ready;
        if(eiglog) {
            eiglog->trace(
                "auto residual correction start check: cheap dwell {}/{} ready {} | eigval sat {} ratio {:.6e} threshold {:.6e} std {:.6e} scale {:.6e} "
                "value {:.6e} hist {} enough {} | rrnorm sat {} ratio {:.6e} threshold {:.6e} std {:.6e} scale {:.6e} value {:.6e} hist {} "
                "enough {} | jd start rrnorm ready {} threshold {:.6e} value {:.6e}",
                auto_residual_correction.dwell, config.auto_min_dwell_iters, jd_start_ready, saturation.eigval.saturated, saturation.eigval.ratio,
                saturation.eigval.threshold, saturation.eigval.stddev, saturation.eigval.scale, saturation.eigval.value, saturation.eigval.history_size,
                saturation.eigval.enough_history, saturation.rnorm.saturated, saturation.rnorm.ratio, saturation.rnorm.threshold, saturation.rnorm.stddev,
                saturation.rnorm.scale, saturation.rnorm.value, saturation.rnorm.history_size, saturation.rnorm.enough_history, jd_start_rnorm_ready,
                config.auto_jd_start_rnorm_threshold, rrnorm);
        }
        if(auto_residual_correction.dwell < config.auto_min_dwell_iters && !jd_start_rnorm_ready) return;

        if(jd_start_ready) {
            auto_residual_correction.active               = ResidualCorrectionType::JACOBI_DAVIDSON;
            auto_residual_correction.dwell                = 0;
            auto_residual_correction.jd_steps_since_probe = 0;
            auto_residual_correction.cheap_to_jd_switch_iters.push_back(status.iter);

            if(eiglog) {
                eiglog->debug(
                    "auto residual correction switch: {} -> {} | reason {} | eigval ratio {:.6e} threshold {:.6e} std {:.6e} scale {:.6e} value {:.6e} | "
                    "rrnorm ratio {:.6e} threshold {:.6e} std {:.6e} scale {:.6e} value {:.6e} | probe interval {} probe factor {:.6e} | "
                    "jd start rrnorm threshold {:.6e} rrnorm {:.6e} | "
                    "mv {} outer {} inner {} inner_iters {} jdops {} time {:.6e}s",
                    method_name(ResidualCorrectionType::CHEAP_OLSEN), method_name(ResidualCorrectionType::JACOBI_DAVIDSON),
                    jd_start_rnorm_ready ? "rrnorm threshold" : "saturation", saturation.eigval.ratio, saturation.eigval.threshold, saturation.eigval.stddev,
                    saturation.eigval.scale, saturation.eigval.value, saturation.rnorm.ratio, saturation.rnorm.threshold, saturation.rnorm.stddev,
                    saturation.rnorm.scale, saturation.rnorm.value, config.auto_cheap_probe_interval, config.auto_cheap_probe_factor,
                    config.auto_jd_start_rnorm_threshold, rrnorm, status.num_matvecs + status.num_matvecs_inner, status.num_matvecs, status.num_matvecs_inner,
                    status.num_iters_inner, status.num_jdops_inner, step_seconds);
            }
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

        auto ProjectOpL = [&V](const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> Y) -> void {
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

        auto ProjectOpR = [&V](const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> Y) -> void {
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
                auto token_precond = status.time_precond_inner.tic_token();
                A.preconditioner_update(th);

                IterativeLinearSolverConfig<Scalar> cfg;
                cfg.result               = {};
                cfg.matdef               = MatDef::IND;
                cfg.maxiters             = config.inner_max_iters;
                cfg.tolerance            = current_inner_tol;
                cfg.theta                = th;
                cfg.preconditioner_apply = [this](const Eigen::Ref<const VectorType> &x, Eigen::Ref<VectorType> y, RealScalar theta) -> void {
                    A.preconditioner_apply(x, y, theta);
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

                status.num_iters_inner   += cfg.result.iters;
                status.num_jdops_inner   += cfg.result.matvecs;
                status.num_precond_inner += cfg.result.precond;
                status.inner_error_last   = std::max(status.inner_error_last, cfg.result.error);
                status.inner_tol_last     = std::max(status.inner_tol_last, cfg.tolerance);
            }
        }
        status.num_precond += cfg().block_size;
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

        auto ProjectOpL = [&V, &BV](const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> Y) -> void {
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

        auto ProjectOpR = [&V, &BV](const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> Y) -> void {
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
                auto token_precond = status.time_precond_inner.tic_token();
                A.preconditioner_update(th);

                IterativeLinearSolverConfig<Scalar> cfg;
                cfg.result               = {};
                cfg.matdef               = MatDef::IND;
                cfg.maxiters             = config.inner_max_iters;
                cfg.tolerance            = current_inner_tol;
                cfg.theta                = th;
                cfg.preconditioner_apply = [this](const Eigen::Ref<const VectorType> &x, Eigen::Ref<VectorType> y, RealScalar theta) -> void {
                    A.preconditioner_apply(x, y, theta);
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

                status.num_iters_inner   += cfg.result.iters;
                status.num_jdops_inner   += cfg.result.matvecs;
                status.num_precond_inner += cfg.result.precond;
                status.inner_error_last   = std::max(status.inner_error_last, cfg.result.error);
                status.inner_tol_last     = std::max(status.inner_tol_last, cfg.tolerance);
            }
        }
        status.num_precond += cfg().block_size;
        return D;
    }

    template<typename Scalar, grit::Form form_>
    typename gdplusk<Scalar, form_>::MatrixType gdplusk<Scalar, form_>::get_sBlock(const MatrixType &S_in) {
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

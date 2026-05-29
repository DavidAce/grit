#pragma once

#include <grit/form/base.h>
#include <stdexcept>

namespace settings {
#if defined(NDEBUG)
    static constexpr bool debug_ortho = false;
#else
    static constexpr bool debug_ortho = true;
#endif
}

namespace grit::form {
    template<typename Scalar, grit::Form form_>
    std::string_view base<Scalar, form_>::ResidualCorrectionToString(ResidualCorrectionType rct) {
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
    typename base<Scalar, form_>::ResidualCorrectionType base<Scalar, form_>::StringToResidualCorrection(std::string_view rct) {
        if(rct == "NONE") return ResidualCorrectionType::NONE;
        if(rct == "CHEAP_OLSEN") return ResidualCorrectionType::CHEAP_OLSEN;
        if(rct == "FULL_OLSEN") return ResidualCorrectionType::FULL_OLSEN;
        if(rct == "JACOBI_DAVIDSON") return ResidualCorrectionType::JACOBI_DAVIDSON;
        if(rct == "AUTO") return ResidualCorrectionType::AUTO;
        return ResidualCorrectionType::NONE;
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::bind_config(BaseConfig &config) {
        cfg_ptr = &config;
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::BaseConfig &base<Scalar, form_>::cfg() {
        return *cfg_ptr;
    }

    template<typename Scalar, grit::Form form_>
    const typename base<Scalar, form_>::BaseConfig &base<Scalar, form_>::cfg() const {
        return *cfg_ptr;
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::setLogger(spdlog::level::level_enum level, const std::string &name) {
        eiglog = Logger::getLogger(name.empty() ? "grit" : name);
        eiglog->set_level(level);
    }

    template<typename Scalar, grit::Form form_>
    base<Scalar, form_>::base(const MatrixType &V, Matvec<Scalar> &A) requires(form_ == grit::Form::STANDARD) : A(A), V(V) {
        setLogger(cfg().log_level, "grit");
        N    = A.get_size();
        size = A.get_size();
        status.rNorms.setOnes(cfg().nev);
        status.eigVal.setOnes(cfg().nev);
        status.oldVal.setOnes(cfg().nev);
        status.absDiff.setOnes(cfg().nev);
        status.relDiff.setOnes(cfg().nev);
    }

    template<typename Scalar, grit::Form form_>
    base<Scalar, form_>::base(const MatrixType &V, Matvec<Scalar> &A, Matvec<Scalar> &B) requires(form_ == grit::Form::GENERALIZED) : A(A), B(B), V(V) {
        setLogger(cfg().log_level, "grit");
        N    = A.get_size();
        size = A.get_size();
        status.rNorms.setOnes(cfg().nev);
        status.eigVal.setOnes(cfg().nev);
        status.oldVal.setOnes(cfg().nev);
        status.absDiff.setOnes(cfg().nev);
        status.relDiff.setOnes(cfg().nev);
    }

    template<typename Scalar, grit::Form form_>
    std::string_view base<Scalar, form_>::form_name() const {
        if constexpr(form_ == grit::Form::GENERALIZED) return "GENERALIZED";
        else return "STANDARD";
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::MatrixType base<Scalar, form_>::get_residuals(const Eigen::Ref<const VectorReal> &Y, const Eigen::Ref<const MatrixType> &AV,
                                                                  const Eigen::Ref<const MatrixType> &BV, VectorReal &rNorms) {
        MatrixType S = AV - BV * Y.asDiagonal();
        rNorms       = S.colwise().norm().transpose();
        return S;
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::RealScalar base<Scalar, form_>::rNormTol([[maybe_unused]] Eigen::Index n) const {
        if(cfg().use_relative_rnorm_tolerance) {
            auto scale     = relative_rNormScale(n);
            auto rnorm_tol = cfg().tol * scale;
            if(cfg().tol_rnorm_relative > RealScalar{0} && n < status.rNorms_init.size())
                rnorm_tol = std::max(rnorm_tol, cfg().tol_rnorm_relative * status.rNorms_init(n));
            return rnorm_tol;
        }
        return cfg().tol;
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::VectorReal base<Scalar, form_>::rNormTols() const {
        VectorReal tols(cfg().nev);
        for(Eigen::Index n = 0; n < cfg().nev; ++n) tols(n) = rNormTol(n);
        return tols;
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::RealScalar base<Scalar, form_>::relative_rNormScale(Eigen::Index n) const {
        auto op_norm = get_op_norm_estimate(status.eigVal.size() > n ? std::optional<RealScalar>{status.eigVal(n)} : std::nullopt);
        auto v_norm  = V.cols() > n ? V.col(n).norm() : RealScalar{1};
        if(!std::isfinite(v_norm) || v_norm <= RealScalar{0}) v_norm = RealScalar{1};
        return std::max(op_norm * v_norm, std::numeric_limits<RealScalar>::min());
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::VectorReal base<Scalar, form_>::relative_rNormScales() const {
        VectorReal scales(cfg().nev);
        for(Eigen::Index n = 0; n < cfg().nev; ++n) scales(n) = relative_rNormScale(n);
        return scales;
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::VectorReal base<Scalar, form_>::relative_rNorms(const VectorReal &rNorms) const {
        auto rows = std::min<Eigen::Index>(cfg().nev, rNorms.size());
        if(rows <= 0) return {};
        return rNorms.topRows(rows).cwiseQuotient(relative_rNormScales().topRows(rows));
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::RealScalar base<Scalar, form_>::get_rNorms_log10_change_per_matvec() {
        if(status.rNorms_history.size() < 2ul) return RealScalar{0};
        auto size = status.rNorms_history.size();
        assert(size == status.matvecs_history.size());

        auto rNorm_change = status.rNorms_history[size - 1].array() / status.rNorms_history[size - 2].array();
        auto sum_matvecs  = status.matvecs_history[size - 1] + status.matvecs_history[size - 2];
        if(sum_matvecs <= 0) return RealScalar{0};
        return std::log10(rNorm_change.minCoeff()) / static_cast<RealScalar>(sum_matvecs);
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::RealScalar base<Scalar, form_>::get_op_norm_estimate(std::optional<RealScalar> eigval) const {
        auto op_norm_estimate = std::max(RealScalar{1}, status.op_norm_estimate);
        if(!std::isfinite(op_norm_estimate)) op_norm_estimate = RealScalar{1};

        auto abs_eigval = std::abs(eigval.value_or(status.eigVal.size() > 0 ? status.eigVal(0) : RealScalar{1}));
        if(status.eigVal.size() > 0) abs_eigval = std::max(abs_eigval, status.eigVal.cwiseAbs().maxCoeff());
        if(T_evals.size() > 0) abs_eigval = std::max(abs_eigval, T_evals.cwiseAbs().maxCoeff());

        if(Q.size() == 0 || Q.norm() == RealScalar{0}) return std::max({op_norm_estimate, abs_eigval, A.get_op_norm()});

        if constexpr(form_ == grit::Form::GENERALIZED) {
            auto A_maxnorm = AQ.size() == Q.size() ? AQ.norm() / Q.norm() : RealScalar{1};
            auto B_maxnorm = BQ.size() == Q.size() ? BQ.norm() / Q.norm() : RealScalar{1};
            return std::max({op_norm_estimate, A_maxnorm + abs_eigval * B_maxnorm, abs_eigval, A.get_op_norm()});
        } else {
            auto A_maxnorm = AQ.size() == Q.size() ? AQ.norm() / Q.norm() : RealScalar{1};
            return std::max({op_norm_estimate, A_maxnorm, abs_eigval, A.get_op_norm()});
        }
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::MatrixType base<Scalar, form_>::MultA(const Eigen::Ref<const MatrixType> &X) {
        auto token_matvecs  = status.time_matvecs.tic_token();
        status.num_matvecs += X.cols();
        return A.mult(X);
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::MatrixType base<Scalar, form_>::MultA_inner(const Eigen::Ref<const MatrixType> &X) {
        auto token_matvecs        = status.time_matvecs_inner.tic_token();
        status.num_matvecs_inner += X.cols();
        return A.mult(X);
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::MatrixType base<Scalar, form_>::MultB(const Eigen::Ref<const MatrixType> &X) requires(form_ == grit::Form::GENERALIZED) {
        auto token_matvecs  = status.time_matvecs.tic_token();
        status.num_matvecs += X.cols();
        return B->get().mult(X);
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::MatrixType base<Scalar, form_>::MultB_inner(const Eigen::Ref<const MatrixType> &X) requires(form_ == grit::Form::GENERALIZED) {
        auto token_matvecs        = status.time_matvecs_inner.tic_token();
        status.num_matvecs_inner += X.cols();
        return B->get().mult(X);
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::MatrixType base<Scalar, form_>::MultP(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const VectorReal> &evals) {
        if(!A.has_preconditioner_apply()) return X;

        auto       token_precond = status.time_precond.tic_token();
        MatrixType Y(X.rows(), X.cols());
        for(Eigen::Index i = 0; i < X.cols(); ++i) {
            RealScalar theta = evals(std::min<Eigen::Index>(i, evals.size() - 1));
            A.preconditioner_update(theta);
            auto x = X.col(i);
            auto y = Y.col(i);
            A.preconditioner_apply(x, y, theta);
        }
        status.num_precond += X.cols();
        return Y;
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::diagonalizeT() {
        T1 = Q.adjoint() * AQ;
        T1 = (T1 + T1.adjoint()) * half;
        if constexpr(form_ == grit::Form::GENERALIZED) {
            T2 = Q.adjoint() * BQ;
            T2 = (T2 + T2.adjoint()) * half;
            Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType> es(T1, T2);
            if(es.info() != Eigen::Success) throw std::runtime_error("diagonalizeT: generalized eigensolver failed");
            T_evals = es.eigenvalues();
            T_evecs = es.eigenvectors();
        } else {
            T2 = MatrixType::Identity(T1.rows(), T1.cols());
            Eigen::SelfAdjointEigenSolver<MatrixType> es(T1);
            if(es.info() != Eigen::Success) throw std::runtime_error("diagonalizeT: eigensolver failed");
            T_evals = es.eigenvalues();
            T_evecs = es.eigenvectors();
        }
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::init() {
        assert(N == A.get_size() && "A must have same dimension");
        status.saturation_count_max = cfg().ncv;
        Eigen::ColPivHouseholderQR<MatrixType> cpqr;

        // Step 0: Construct and orthonormalize the initial block V.
        // We aim to construct V = [v[0]...v[block_size-1]], where v are ritz eigenvectors.
        // If V has fewer than block_size columns, we pad it with random vectors and orthonormalize with ColPivHouseholderQR.
        // If V has more than block_size columns, we discard the overshooting columns after QR.
        // If after QR we have fewer than block_size columns, we pad again (this is a very unlikely event)
        assert(V.size() == 0 or N == V.rows());
        for(long i = 0; i < 2; ++i) {
            if(V.cols() < cfg().block_size) {
                // Pad with random vectors
                auto vc = V.cols();
                V.conservativeResize(N, cfg().block_size);
                auto Vrc = V.rightCols(cfg().block_size - vc);
                for(auto vj : Vrc.colwise()) { vj = Eigen::VectorXf::Random(vj.size()).template cast<Scalar>(); }
            }
            // Orthonormalize V.
            // Discard columns if there are more than block_size (this is not expected, but also not an error)
            cpqr.compute(V);
            auto rank = std::min<Eigen::Index>(cpqr.rank(), cfg().block_size);
            V         = cpqr.householderQ().setLength(rank) * MatrixType::Identity(N, rank);
            if(V.cols() == cfg().block_size) break;
        }

        auto block_orthonormalize = [&] {
            auto m       = OrthMeta();
            m.maskPolicy = MaskPolicy::COMPRESS;
            if constexpr(form_ == grit::Form::GENERALIZED) {
                if(cfg().use_b_inner_product) {
                    block_bm_orthonormalize(V, AV, BV, m);
                } else {
                    block_l2_orthonormalize(V, AV, BV, m);
                }
            } else {
                block_l2_orthonormalize(V, AV, m);
            }
        };

        assert(V.cols() == cfg().block_size);
        if(status.iter == 0) {
            // Make sure we start with ritz vectors in V, so that the first Lanczos loop produces proper residuals.
            if constexpr(form_ == grit::Form::GENERALIZED) {
                block_orthonormalize();
                Q             = V;
                AQ            = AV;
                BQ            = BV;
                MatrixType T1 = Q.adjoint() * AQ;
                MatrixType T2 = Q.adjoint() * BQ;
                T1            = RealScalar{0.5f} * (T1.adjoint() + T1); // Symmetrize
                T2            = RealScalar{0.5f} * (T2.adjoint() + T2); // Symmetrize
                Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType> es_seed(T1, T2, Eigen::Ax_lBx);
                T_evecs       = es_seed.eigenvectors();
                T_evals       = es_seed.eigenvalues();
                status.optIdx = get_ritz_indices(cfg().ritz, 0, cfg().block_size, T_evals);
                MatrixType Z  = T_evecs(Eigen::placeholders::all, status.optIdx);
                VectorReal Y  = T_evals(status.optIdx);
                V             = Q * Z;  // Now V has block_size columns mixed according to the selected columns in T_evecs
                AV            = AQ * Z; // Now AV has block_size columns mixed according to the selected columns in T_evecs
                BV            = BQ * Z; // Now BV has block_size columns mixed according to the selected columns in T_evecs

                RealScalar min_sep =
                    T_evals.size() <= 1 ? RealScalar{1} : (T_evals.tail(T_evals.size() - 1) - T_evals.head(T_evals.size() - 1)).cwiseAbs().minCoeff();
                auto select1       = get_ritz_indices(cfg().ritz, 0, 1, T_evals);
                auto A_max_abs     = std::max({T1.cwiseAbs().maxCoeff(), AV.norm() / V.norm(), RealScalar{1}});
                auto B_max_abs     = std::max({T2.cwiseAbs().maxCoeff(), BV.norm() / V.norm(), RealScalar{1}});
                status.sensitivity = (A_max_abs + T_evals(select1).cwiseAbs().coeff(0) * B_max_abs) / min_sep;

                auto AB_max_abs         = T_evals.cwiseAbs().maxCoeff();
                auto AB_min_abs         = T_evals.cwiseAbs().minCoeff();
                status.condition        = AB_max_abs / std::max(AB_min_abs, std::numeric_limits<RealScalar>::min());
                status.op_norm_estimate = A_max_abs + T_evals(select1).cwiseAbs().coeff(0) * B_max_abs;
                // We may need to orthonormalize V in generalized problems
                block_orthonormalize();

                S             = get_residuals(Y, AV, BV, status.rNorms);
                status.eigVal = Y.topRows(cfg().nev); // Make sure we only take nev values here. In general, nev <= block_size
            } else {
                block_orthonormalize();
                Q  = V;
                AQ = AV;
                T  = Q.adjoint() * AQ;
                T  = RealScalar{0.5f} * (T.adjoint() + T); // Symmetrize
                Eigen::SelfAdjointEigenSolver<MatrixType> es(T);
                T_evecs       = es.eigenvectors();
                T_evals       = es.eigenvalues();
                status.optIdx = get_ritz_indices(cfg().ritz, 0, cfg().block_size, T_evals);
                MatrixType Z  = T_evecs(Eigen::placeholders::all, status.optIdx);
                VectorReal Y  = T_evals(status.optIdx);
                V             = Q * Z; // Now V has block_size columns mixed according to the selected columns in T_evecs
                AV            = AQ * Z;
                BV            = V;
                BQ            = Q;
                S             = get_residuals(Y, AV, V, status.rNorms);
                status.eigVal = Y.topRows(cfg().nev); // Make sure we only take nev values here. In general, nev <= block_size

                auto A_max_abs          = T_evals.cwiseAbs().maxCoeff();
                auto A_min_abs          = T_evals.cwiseAbs().minCoeff();
                status.condition        = A_max_abs / std::max(A_min_abs, std::numeric_limits<RealScalar>::min());
                status.op_norm_estimate = std::max({A_max_abs, AQ.norm() / Q.norm(), RealScalar{1}});
            }
        }
        status.rNorms_init      = status.rNorms;
        status.op_norm_estimate = get_op_norm_estimate();
        assert(V.cols() == cfg().block_size);
        assert_allFinite(V);
        last_log_time.tic();
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::preamble() {
        status.num_iters_inner_prev = status.num_iters_inner;
        status.num_matvecs          = 0;
        status.num_precond          = 0;
        status.num_iters_inner      = 0;
        status.num_matvecs_inner    = 0;
        status.num_precond_inner    = 0;
        status.num_jdops_inner      = 0;

        status.inner_error_last = RealScalar{0};
        status.inner_tol_last   = RealScalar{0};

        status.time_matvecs.reset();
        status.time_precond.reset();
        status.time_matvecs_inner.reset();
        status.time_precond_inner.reset();
        status.time_jdops_inner.reset();
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::VectorReal base<Scalar, form_>::get_standard_deviations(const std::deque<VectorReal> &v, bool apply_log10) {
        if(v.empty()) return {};
        auto       cols         = static_cast<Eigen::Index>(v.size());
        auto       rows         = static_cast<Eigen::Index>(v.front().size());
        MatrixReal matrix       = MatrixReal::Zero(rows, cols);
        using history_size_type = typename std::deque<VectorReal>::size_type;
        for(Eigen::Index idx = 0; idx < cols; ++idx) {
            auto udx = static_cast<history_size_type>(idx);
            if(v[udx].size() < rows) throw std::runtime_error("v has unequal size vectors");
            if(apply_log10)
                matrix.col(idx) = v[udx].topRows(rows).array().log10();
            else
                matrix.col(idx) = v[udx].topRows(rows).array();
        }
        VectorReal means = matrix.rowwise().mean();
        if(matrix.cols() <= 1) return VectorReal::Zero(rows);
        VectorReal stddev = (((matrix.colwise() - means).array().square().rowwise().sum()) / static_cast<RealScalar>((matrix.cols() - 1))).sqrt();
        return stddev;
    }

    template<typename Scalar, grit::Form form_>
    bool base<Scalar, form_>::rNorms_have_saturated() {
        if(cfg().sat_rnorm_threshold <= RealScalar{0}) return false;
        const auto min_history_size = std::min<size_t>(status.max_history_size, size_t{2});
        if(status.iter < static_cast<Eigen::Index>(min_history_size)) return false;
        if(status.rNorms_history.size() < static_cast<size_t>(min_history_size)) return false;
        if(status.rNorms.size() == 0) return false;

        auto rows = std::min<Eigen::Index>(cfg().nev, status.rNorms.size());
        if(rows <= 0) return false;
        VectorReal             scales = relative_rNormScales().topRows(rows);
        std::deque<VectorReal> relative_history;
        for(const auto &history : status.rNorms_history) {
            if(history.size() < rows) throw std::runtime_error("rNorms_history has unequal size vectors");
            relative_history.emplace_back(history.topRows(rows).cwiseQuotient(scales));
        }
        VectorReal vals           = status.rNorms.topRows(rows).cwiseQuotient(scales);
        VectorReal stds           = get_standard_deviations(relative_history, false);
        VectorReal threshold      = cfg().sat_rnorm_threshold * vals.cwiseMax(VectorReal::Constant(vals.size(), std::numeric_limits<RealScalar>::min()));
        VectorIdxT stds_saturated = (stds.array() < threshold.array()).template cast<Eigen::Index>();
        return stds_saturated.all();
    }

    template<typename Scalar, grit::Form form_>
    bool base<Scalar, form_>::eigVals_have_saturated() {
        if(cfg().sat_eigval_threshold <= RealScalar{0}) return false;
        const auto min_history_size = std::min<size_t>(status.max_history_size, size_t{2});
        if(status.iter < static_cast<Eigen::Index>(min_history_size)) return false;
        if(status.eigVals_history.size() < static_cast<size_t>(min_history_size)) return false;
        VectorReal vals           = status.eigVal.cwiseAbs().array() + eps;
        VectorReal stds           = get_standard_deviations(status.eigVals_history, false);
        VectorReal rels           = stds.cwiseQuotient(vals);
        VectorIdxT rels_saturated = (rels.array() < cfg().sat_eigval_threshold).template cast<Eigen::Index>();
        return rels_saturated.all();
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::updateStatus() {
        // Accumulate counters from the inner solver
        status.num_matvecs_total  += status.num_matvecs + status.num_matvecs_inner;
        status.num_precond_total  += status.num_precond + status.num_precond_inner;
        status.time_matvecs_total += status.time_matvecs.get_time() + status.time_matvecs_inner.get_time();
        status.time_precond_total += status.time_precond.get_time() + status.time_precond_inner.get_time();

        // Eigenvalues are sorted in ascending order.
        status.oldVal  = status.eigVal.topRows(cfg().nev);
        status.eigVal  = T_evals(status.optIdx).topRows(cfg().nev);
        status.absDiff = (status.eigVal - status.oldVal).cwiseAbs();

        VectorReal denom = (RealScalar{0.5} * (status.eigVal + status.oldVal).array().abs()).matrix();
        denom            = denom.cwiseMax(VectorReal::Constant(denom.size(), std::numeric_limits<RealScalar>::min()));
        status.relDiff   = status.absDiff.cwiseQuotient(denom);

        status.op_norm_estimate = get_op_norm_estimate();

        status.rNorms_history.push_back(status.rNorms.topRows(cfg().nev));
        status.eigVals_history.push_back(status.eigVal.topRows(cfg().nev));
        status.matvecs_history.push_back(status.num_matvecs + status.num_matvecs_inner);
        while(status.rNorms_history.size() > status.max_history_size) status.rNorms_history.pop_front();
        while(status.eigVals_history.size() > status.max_history_size) status.eigVals_history.pop_front();
        while(status.matvecs_history.size() > status.max_history_size) status.matvecs_history.pop_front();

        if(eigVals_have_saturated())
            status.saturation_count_eigVal++;
        else
            status.saturation_count_eigVal = 0;

        if(rNorms_have_saturated())
            status.saturation_count_rNorm++;
        else
            status.saturation_count_rNorm = 0;

        auto format_vector = [](const VectorReal &v, std::string_view spec = "{:.3e}") {
            std::string msg = "[";
            for(Eigen::Index i = 0; i < v.size(); ++i) {
                if(i > 0) msg += ", ";
                msg += fmt::format(fmt::runtime(std::string(spec)), v(i));
            }
            msg += "]";
            return msg;
        };

        constexpr auto beta    = RealScalar{0.5f};
        VectorReal     rNorms  = status.rNorms.topRows(cfg().nev);
        VectorReal     rrNorms = relative_rNorms(status.rNorms).topRows(cfg().nev);
        VectorReal     tols    = rNormTols();
        RealScalar     relGap  = status.gap * get_op_norm_estimate(status.eigVal.size() > 0 ? std::optional<RealScalar>{status.eigVal(0)} : std::nullopt);
        if(rNorms.size() != tols.size()) throw std::logic_error("unequal residual norm and tolerance sizes");
        status.rNorm_below_rnormTol = (rNorms.array() < tols.array()).all();
        status.rNorm_below_gap      = rNorms.maxCoeff() < beta * relGap;

        if(status.rNorm_below_rnormTol) {
            std::string msg_rnorm_gap = fmt::format(" | gap {:.3e} (rel {:.3e})", status.gap, relGap);
            if(cfg().use_relative_rnorm_tolerance) {
                status.stopMessage.emplace_back(fmt::format("converged rNorm {} < relative tol {}{} | rrNorm {} | iters {} | mv {} | {:.3e} s",
                                                            format_vector(VectorReal(status.rNorms.topRows(cfg().nev))), format_vector(tols), msg_rnorm_gap,
                                                            format_vector(rrNorms), status.iter + 1, status.num_matvecs_total, status.time_elapsed.get_time()));
            } else {
                status.stopMessage.emplace_back(fmt::format("converged rNorm {} < tol {}{} | rrNorm {} | iters {} | mv {} | {:.3e} s",
                                                            format_vector(VectorReal(status.rNorms.topRows(cfg().nev))), format_vector(tols), msg_rnorm_gap,
                                                            format_vector(rrNorms), status.iter + 1, status.num_matvecs_total, status.time_elapsed.get_time()));
            }
            status.stopReason |= StopReason::converged;
        }

        if(cfg().max_iters >= 0 && status.iter + 1 >= cfg().max_iters) {
            status.stopMessage.emplace_back(fmt::format("iters ({}) >= maxiter ({}) | mv {} | {:.3e} s", status.iter + 1, cfg().max_iters,
                                                        status.num_matvecs_total, status.time_elapsed.get_time()));
            status.stopReason |= StopReason::max_iters;
        }

        if(cfg().max_matvecs >= 0 && status.num_matvecs_total >= cfg().max_matvecs) {
            status.stopMessage.emplace_back(fmt::format("num_matvecs_total ({}) >= max_matvecs ({}) | {:.3e} s", status.num_matvecs_total, cfg().max_matvecs,
                                                        status.time_elapsed.get_time()));
            status.stopReason |= StopReason::max_matvecs;
        }

        if(std::min(status.saturation_count_eigVal, status.saturation_count_rNorm) >= status.saturation_count_max) {
            status.stopMessage.emplace_back(fmt::format("saturation_count (eigVal {} rNorm {}) >= saturation_count_max ({}) | it {} | mv {} | {:.3e} s",
                                                        status.saturation_count_eigVal, status.saturation_count_rNorm, status.saturation_count_max,
                                                        status.iter + 1, status.num_matvecs_total, status.time_elapsed.get_time()));
            status.stopReason |= StopReason::ritz_value_stalled;
            status.stopReason |= StopReason::ritz_residual_stalled;
        } else if(status.saturation_count_eigVal >= status.saturation_count_max * 2) {
            status.stopMessage.emplace_back(fmt::format("saturation_count eigVal {} >= saturation_count_max ({}) * 2 | it {} | mv {} | {:.3e} s",
                                                        status.saturation_count_eigVal, status.saturation_count_max, status.iter + 1, status.num_matvecs_total,
                                                        status.time_elapsed.get_time()));
            status.stopReason |= StopReason::ritz_value_stalled;
        } else if(status.saturation_count_eigVal > 2 && status.saturation_count_rNorm >= status.saturation_count_max * 2) {
            // Probably eigVal is stuck in some kind of cycle.
            status.stopMessage.emplace_back(fmt::format("saturation_count rNorm {} >= saturation_count_max ({}) * 2 | it {} | mv {} | {:.3e} s",
                                                        status.saturation_count_rNorm, status.saturation_count_max, status.iter + 1, status.num_matvecs_total,
                                                        status.time_elapsed.get_time()));
            status.stopReason |= StopReason::ritz_residual_stalled;
        }
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::printStatus() {
        if(!eiglog || status.eigVal.size() == 0 || status.rNorms.size() == 0) return;

        auto format_vector = [](const VectorReal &v, std::string_view spec = "{:.8e}") {
            std::string msg = "[";
            for(Eigen::Index i = 0; i < v.size(); ++i) {
                if(i > 0) msg += ", ";
                msg += fmt::format(fmt::runtime(std::string(spec)), v(i));
            }
            msg += "]";
            return msg;
        };

        std::string msg_rnorm_gap = fmt::format(" | gap {:.3e}", status.gap);

        std::string rCorrMsg;
        switch(residual_correction_type_internal) {
            case ResidualCorrectionType::NONE: rCorrMsg = "NO"; break;
            case ResidualCorrectionType::CHEAP_OLSEN: rCorrMsg = "CO"; break;
            case ResidualCorrectionType::FULL_OLSEN: rCorrMsg = "FO"; break;
            case ResidualCorrectionType::JACOBI_DAVIDSON: rCorrMsg = "JD"; break;
            case ResidualCorrectionType::AUTO: rCorrMsg = "AU"; break;
        }

        std::string innerMsg;
        if(status.num_matvecs_inner > 0 || status.num_jdops_inner > 0 || status.num_precond_inner > 0) {
            innerMsg = fmt::format("[inner: ({}) mv {:5} jd {:5} pc {:5} err {:.2e} tol {:.2e} mv {:.1e}s jd {:.1e}s pc {:.1e}s] ", rCorrMsg,
                                   status.num_matvecs_inner, status.num_jdops_inner, status.num_precond_inner, status.inner_error_last, status.inner_tol_last,
                                   status.time_matvecs_inner.get_time(), status.time_jdops_inner.get_time(), status.time_precond_inner.get_time());
        }

        RealScalar orthError = RealScalar{0};
        if(Q.cols() > 0) {
            MatrixType Gram = cfg().use_b_inner_product && BQ.size() == Q.size() ? Q.adjoint() * BQ : Q.adjoint() * Q;
            Gram            = (Gram + Gram.adjoint()).eval() / RealScalar{2};
            orthError       = (Gram - MatrixType::Identity(Gram.rows(), Gram.cols())).norm();
        }

        std::string evMsg;
        if constexpr(form_ == grit::Form::GENERALIZED) {
            if(V.cols() > 0 && AV.size() == V.size() && BV.size() == V.size()) {
            VectorReal VAV = (V.adjoint() * AV).diagonal().real();
            VectorReal VBV = (V.adjoint() * BV).diagonal().real();
            evMsg          = fmt::format(" {} / {}", format_vector(VAV, "{:.16f}"), format_vector(VBV, "{:.16f}"));
            }
        }

        auto op_norm_estimate              = get_op_norm_estimate(status.eigVal.size() > 0 ? std::optional<RealScalar>{status.eigVal(0)} : std::nullopt);
        bool log_long_time                 = last_log_time.get_lap() > 10.0;
        bool log_every_ten_it              = status.iter > 0 && status.iter % 10 == 0;
        spdlog::level::level_enum loglevel = spdlog::level::trace;
        if(log_every_ten_it || log_long_time) loglevel = spdlog::level::debug;
        if(!eiglog->should_log(loglevel)) return;
        [[maybe_unused]] auto lap = last_log_time.restart_lap();

        const auto active_tols     = rNormTols();
        const auto active_tol_name = "rNormTol";
        const auto rrNorms         = relative_rNorms(status.rNorms);

        eiglog->log(loglevel,
                    "it {:3} mv {:3} pc {:3} t {:.1e}s dim {} {}eigVal {}{} "
                    "oErr {:.3e} rNorms {} rrNorms {} {} {} tol {:.2e} (rel {:.2e}) "
                    "({:9.2e}/mv) sat {}:{}/{} col {:2} block_size {} ritz {} "
                    "op norm {:.2e} cond {:.2e} sens {:.2e}{}",
                    status.iter, status.num_matvecs, status.num_precond, status.time_elapsed.get_time(), N, innerMsg,
                    format_vector(VectorReal(status.eigVal), "{:.16f}"), evMsg, orthError, format_vector(VectorReal(status.rNorms)),
                    format_vector(VectorReal(rrNorms)), active_tol_name, format_vector(active_tols, "{:.3e}"), cfg().tol, cfg().tol_rnorm_relative,
                    get_rNorms_log10_change_per_matvec(), status.saturation_count_eigVal, status.saturation_count_rNorm, status.saturation_count_max, Q.cols(),
                    cfg().block_size, enum2sv(cfg().ritz), op_norm_estimate, status.condition, status.sensitivity, msg_rnorm_gap);
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::run_user_callback() {
        return;
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::clear_result() {
        status                   = Status{};
        qBlocks                  = 0;
        T                        = MatrixType{};
        Aproj                    = MatrixType{};
        Bproj                    = MatrixType{};
        W                        = MatrixType{};
        Q                        = MatrixType{};
        AQ                       = MatrixType{};
        BQ                       = MatrixType{};
        V                        = MatrixType{};
        AV                       = MatrixType{};
        BV                       = MatrixType{};
        V_prev                   = MatrixType{};
        K                        = MatrixType{};
        K_prev                   = MatrixType{};
        S                        = MatrixType{};
        S1                       = MatrixType{};
        S2                       = MatrixType{};
        D                        = MatrixType{};
        M                        = MatrixType{};
        AM                       = MatrixType{};
        BM                       = MatrixType{};
        T_evals                  = VectorReal{};
        T1                       = MatrixType{};
        T2                       = MatrixType{};
        T_evecs                  = MatrixType{};
        hhqr                     = Eigen::HouseholderQR<MatrixType>{};
        auto_residual_correction = AutoResidualCorrectionState{};
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::step() {
        preamble();
        build();
        diagonalizeT();
        extractRitzVectors();
        updateStatus();
        printStatus();
        run_user_callback();
        status.iter++;
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::run() {
        auto token_elapsed = status.time_elapsed.tic_token();
        init();
        printStatus();
        run_user_callback();
        status.iter++;
        while(true) {
            step();
            if(status.stopReason != StopReason::none) break;
        }
    }

}

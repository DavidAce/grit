#pragma once

#include "grit/form/base.h"
#include <stdexcept>

namespace grit::settings {
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
        log = Logger::getLogger(name.empty() ? "grit" : name);
        log->set_level(level);
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::set_initial_guess(MatrixType guess) {
        V = std::move(guess);
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::clear_initial_guess() {
        V.resize(0, 0);
    }

    template<typename Scalar, grit::Form form_>
    bool base<Scalar, form_>::has_initial_guess() const {
        return V.size() > 0;
    }

    template<typename Scalar, grit::Form form_>
    const typename base<Scalar, form_>::MatrixType &base<Scalar, form_>::initial_guess() const {
        return V;
    }

    template<typename Scalar, grit::Form form_>
    base<Scalar, form_>::base(const MatrixType &V, Matvec<Scalar> &A) requires(form_ == grit::Form::STANDARD)
        : A(A), V(V) {
        setLogger(cfg().log_level, "grit");
        N    = A.get_size();
        size = A.get_size();
        status.rNormsAbs.setOnes(cfg().nev);
        status.eigVal.setOnes(cfg().nev);
        status.oldVal.setOnes(cfg().nev);
        status.absDiff.setOnes(cfg().nev);
        status.relDiff.setOnes(cfg().nev);
    }

    template<typename Scalar, grit::Form form_>
    base<Scalar, form_>::base(const MatrixType &V, Matvec<Scalar> &A, Matvec<Scalar> &B) requires(form_ == grit::Form::GENERALIZED)
        : A(A), B(B), V(V) {
        setLogger(cfg().log_level, "grit");
        N    = A.get_size();
        size = A.get_size();
        status.rNormsAbs.setOnes(cfg().nev);
        status.eigVal.setOnes(cfg().nev);
        status.oldVal.setOnes(cfg().nev);
        status.absDiff.setOnes(cfg().nev);
        status.relDiff.setOnes(cfg().nev);
    }

    template<typename Scalar, grit::Form form_>
    std::string_view base<Scalar, form_>::form_name() const {
        if constexpr(form_ == grit::Form::GENERALIZED)
            return "GENERALIZED";
        else
            return "STANDARD";
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::MatrixType base<Scalar, form_>::get_residuals(const Eigen::Ref<const VectorReal> &Y, const Eigen::Ref<const MatrixType> &AV,
                                                                                const Eigen::Ref<const MatrixType> &BV, VectorReal &rNormsAbs) {
        MatrixType S = AV - BV * Y.asDiagonal();
        rNormsAbs    = S.colwise().norm().transpose();
        return S;
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::RealScalar base<Scalar, form_>::rNormAbsTarget([[maybe_unused]] Eigen::Index n) const {
        return rNormAbsTargets()(n);
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::VectorReal base<Scalar, form_>::rNormAbsTargets() const {
        VectorReal rNormAbsTargets = VectorReal::Constant(cfg().nev, cfg().abstol);
        if(cfg().use_rescaled_rnorm_tolerance) rNormAbsTargets = rNormAbsTargets.cwiseProduct(rNormScales());

        Eigen::Index rows = std::min(cfg().nev, status.rnorm_abs_reference.size());
        if(cfg().reltol > RealScalar{0} && rows > 0) {
            rNormAbsTargets.topRows(rows) = rNormAbsTargets.topRows(rows).cwiseMax((cfg().reltol * status.rnorm_abs_reference.topRows(rows)).eval());
        }
        return rNormAbsTargets;
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::RealScalar base<Scalar, form_>::rNormScale(Eigen::Index n) const {
        auto op_norm = get_op_norm_estimate(status.eigVal.size() > n ? std::optional<RealScalar>{status.eigVal(n)} : std::nullopt);
        auto v_norm  = V.cols() > n ? V.col(n).norm() : RealScalar{1};
        if(!std::isfinite(v_norm) || v_norm <= RealScalar{0}) v_norm = RealScalar{1};
        return std::max(op_norm * v_norm, std::numeric_limits<RealScalar>::min());
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::VectorReal base<Scalar, form_>::rNormScales() const {
        const auto nev = cfg().nev;
        VectorReal scales(nev);
        for(Eigen::Index n = 0; n < nev; ++n) scales(n) = rNormScale(n);
        return scales;
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::VectorReal base<Scalar, form_>::rNormsRel(const VectorReal &rNormsAbs) const {
        auto rows = std::min<Eigen::Index>(cfg().nev, rNormsAbs.size());
        if(rows <= 0) return {};
        return rNormsAbs.topRows(rows).cwiseQuotient(rNormScales().topRows(rows));
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::RealScalar base<Scalar, form_>::get_rNorms_log10_change_per_matvec() {
        if(status.rNormsAbsHistory.size() < 2ul) return RealScalar{0};
        auto size = status.rNormsAbsHistory.size();
        assert(size == status.matvecs_history.size());

        auto rNorm_change = status.rNormsAbsHistory[size - 1].array() / status.rNormsAbsHistory[size - 2].array();
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
        auto t_matvecs_a      = status.time_matvecs_a.tic_token();
        auto t_matvecs        = status.time_matvecs.tic_token();
        status.num_matvecs_a += X.cols();
        status.num_matvecs   += X.cols();
        return A.mult(X);
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::MatrixType base<Scalar, form_>::MultA_inner(const Eigen::Ref<const MatrixType> &X) {
        auto t_matvecs_a            = status.time_matvecs_a_inner.tic_token();
        status.num_matvecs_a_inner += X.cols();
        status.num_matvecs_inner   += X.cols();
        return A.mult(X);
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::MatrixType base<Scalar, form_>::MultB(const Eigen::Ref<const MatrixType> &X) requires(form_ == grit::Form::GENERALIZED)
    {
        auto t_matvecs_b      = status.time_matvecs_b.tic_token();
        auto t_matvecs        = status.time_matvecs.tic_token();
        status.num_matvecs_b += X.cols();
        status.num_matvecs   += X.cols();
        return B->get().mult(X);
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::MatrixType base<Scalar, form_>::MultB_inner(const Eigen::Ref<const MatrixType> &X) requires(form_ == grit::Form::GENERALIZED)
    {
        auto t_matvecs_b            = status.time_matvecs_b_inner.tic_token();
        status.num_matvecs_b_inner += X.cols();
        status.num_matvecs_inner   += X.cols();
        return B->get().mult(X);
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::MatrixType base<Scalar, form_>::MultP(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const VectorReal> &evals) {
        if(!A.has_preconditioner_apply()) return X;

        MatrixType Y(X.rows(), X.cols());
        for(Eigen::Index i = 0; i < X.cols(); ++i) {
            RealScalar theta = evals(std::min<Eigen::Index>(i, evals.size() - 1));
            if(A.has_preconditioner_update()) {
                A.preconditioner_update(theta);
                status.time_preconditioner_update += A.t_precond_update->get_last_interval();
                status.num_preconditioner_updates++;
            }
            auto x = X.col(i);
            auto y = Y.col(i);
            A.preconditioner_apply(x, y, theta);
            status.time_precond += A.t_precond->get_last_interval();
        }
        status.num_precond += X.cols();
        return Y;
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::diagonalizeT() {
        auto token_diagonalize = status.time_diagonalize.tic_token();
        T1                     = Q.adjoint() * AQ;
        T1                     = (T1 + T1.adjoint()) * half;
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
        status.max_history_size     = static_cast<size_t>(std::max<Eigen::Index>(12, 2 * cfg().ncv / cfg().block_size + 1));
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
        if(status.outer_iter == 0) {
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

                auto select1            = get_ritz_indices(cfg().ritz, 0, 1, T_evals);
                auto A_max_abs          = std::max({T1.cwiseAbs().maxCoeff(), AV.norm() / V.norm(), RealScalar{1}});
                auto B_max_abs          = std::max({T2.cwiseAbs().maxCoeff(), BV.norm() / V.norm(), RealScalar{1}});
                status.op_norm_estimate = A_max_abs + T_evals(select1).cwiseAbs().coeff(0) * B_max_abs;
                // We may need to orthonormalize V in generalized problems
                block_orthonormalize();

                S             = get_residuals(Y, AV, BV, status.rNormsAbs);
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
                S             = get_residuals(Y, AV, V, status.rNormsAbs);
                status.eigVal = Y.topRows(cfg().nev); // Make sure we only take nev values here. In general, nev <= block_size

                auto A_max_abs          = T_evals.cwiseAbs().maxCoeff();
                status.op_norm_estimate = std::max({A_max_abs, AQ.norm() / Q.norm(), RealScalar{1}});
            }
        }
        update_condition_numbers();
        status.op_norm_estimate     = get_op_norm_estimate();
        Eigen::Index rows           = std::min(cfg().nev, status.rNormsAbs.size());
        status.rnorm_abs_reference  = VectorReal::Zero(rows);
        status.num_matvecs_total   += status.num_matvecs;
        status.num_matvecs_a_total += status.num_matvecs_a;
        status.num_matvecs_b_total += status.num_matvecs_b;
        assert(V.cols() == cfg().block_size);
        assert_allFinite(V);
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::preamble() {
        status.num_inner_iters_prev             = status.num_inner_iters;
        status.num_matvecs                      = 0;
        status.num_precond                      = 0;
        status.num_inner_iters                  = 0;
        status.num_matvecs_inner                = 0;
        status.num_matvecs_a                    = 0;
        status.num_matvecs_b                    = 0;
        status.num_matvecs_a_inner              = 0;
        status.num_matvecs_b_inner              = 0;
        status.num_precond_inner                = 0;
        status.num_operator_inner               = 0;
        status.num_preconditioner_updates       = 0;
        status.num_preconditioner_updates_inner = 0;
        status.num_preconditioner_apply_inner   = 0;

        status.inner_error_last = RealScalar{0};
        status.inner_tol_last   = RealScalar{0};
    }

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::VectorReal base<Scalar, form_>::get_standard_deviations(const std::deque<VectorReal> &v, bool apply_log10,
                                                                                           Eigen::Index last_n) {
        if(v.empty()) return {};
        auto       cols         = last_n < 0 ? static_cast<Eigen::Index>(v.size()) : std::min(last_n, static_cast<Eigen::Index>(v.size()));
        auto       rows         = static_cast<Eigen::Index>(v.front().size());
        MatrixReal matrix       = MatrixReal::Zero(rows, cols);
        auto       offset       = static_cast<Eigen::Index>(v.size()) - cols;
        using history_size_type = typename std::deque<VectorReal>::size_type;
        for(Eigen::Index idx = 0; idx < cols; ++idx) {
            auto udx = static_cast<history_size_type>(offset + idx);
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
    typename base<Scalar, form_>::VectorReal base<Scalar, form_>::get_slopes(const std::deque<VectorReal> &v, bool apply_log10, Eigen::Index last_n,
                                                                             VectorReal *slope_errors) {
        if(v.empty()) {
            if(slope_errors) slope_errors->resize(0);
            return {};
        }
        auto rows = static_cast<Eigen::Index>(v.front().size());
        auto cols = last_n < 0 ? static_cast<Eigen::Index>(v.size()) : std::min(last_n, static_cast<Eigen::Index>(v.size()));
        if(cols < 2) {
            if(slope_errors) *slope_errors = VectorReal::Constant(rows, std::numeric_limits<RealScalar>::quiet_NaN());
            return VectorReal::Constant(rows, std::numeric_limits<RealScalar>::quiet_NaN());
        }

        MatrixReal matrix(rows, cols);
        VectorReal x  = VectorReal::LinSpaced(cols, RealScalar{0}, static_cast<RealScalar>(cols - 1));
        x.array()    -= x.mean();
        auto offset   = static_cast<Eigen::Index>(v.size()) - cols;
        for(Eigen::Index idx = 0; idx < cols; ++idx) {
            const auto &history = v[static_cast<typename std::deque<VectorReal>::size_type>(offset + idx)];
            if(history.size() < rows) throw std::runtime_error("history has unequal size vectors");
            if(apply_log10)
                matrix.col(idx) = history.topRows(rows).array().log10();
            else
                matrix.col(idx) = history.topRows(rows);
        }
        auto       x_squared_norm = x.squaredNorm();
        VectorReal slopes         = matrix * x / x_squared_norm;
        if(slope_errors) {
            if(cols < 3) {
                *slope_errors = VectorReal::Constant(rows, std::numeric_limits<RealScalar>::quiet_NaN());
            } else {
                MatrixReal residuals  = matrix.colwise() - matrix.rowwise().mean();
                residuals.noalias()  -= slopes * x.transpose();
                *slope_errors         = (residuals.rowwise().squaredNorm() / (static_cast<RealScalar>(cols - 2) * x_squared_norm)).array().sqrt();
            }
        }
        return slopes;
    }

    template<typename Scalar, grit::Form form_>
    bool base<Scalar, form_>::rNorms_have_saturated() {
        const bool history_ready = status.rNormsAbsHistory.size() >= status.max_history_size;
        const bool have_residuals = status.rNormsAbs.size() > 0;
        if(!history_ready || !have_residuals) return false;

        auto       rows = std::min<Eigen::Index>(cfg().nev, status.rNormsAbs.size());
        VectorReal errors;
        VectorReal slopes = get_slopes(status.rNormsAbsHistory, true, static_cast<Eigen::Index>(status.max_history_size), &errors).topRows(rows);
        VectorReal targets = rNormAbsTargets().topRows(rows);
        bool       saturated = false;
        for(Eigen::Index i = 0; i < rows; ++i) {
            const bool within_target = status.rNormsAbs(i) <= targets(i);
            if(within_target) continue;

            const bool finite_fit           = std::isfinite(slopes(i)) && std::isfinite(errors(i));
            const bool decreasing           = slopes(i) < RealScalar{0};
            const bool significant_decrease = -slopes(i) > RealScalar{2} * errors(i);
            const bool progressing           = finite_fit && decreasing && significant_decrease;
            if(progressing) return false;
            saturated = true;
        }
        return saturated;
    }

    template<typename Scalar, grit::Form form_>
    bool base<Scalar, form_>::eigVals_have_saturated() {
        const bool history_ready = status.eigVals_history.size() >= status.max_history_size;
        const bool have_residuals = status.rNormsAbs.size() > 0;
        if(!history_ready || !have_residuals) return false;

        auto       rows = std::min({cfg().nev, status.eigVal.size(), status.rNormsAbs.size(), V.cols()});
        VectorReal errors;
        VectorReal slopes = get_slopes(status.eigVals_history, false, static_cast<Eigen::Index>(status.max_history_size), &errors).topRows(rows);
        VectorReal targets = rNormAbsTargets().topRows(rows);
        bool       saturated = false;
        for(Eigen::Index i = 0; i < rows; ++i) {
            const bool within_target = status.rNormsAbs(i) <= targets(i);
            if(within_target) continue;

            RealScalar metric_norm = V.col(i).norm();
            if constexpr(form_ == grit::Form::GENERALIZED) metric_norm = BV.col(i).norm();
            RealScalar scaled = std::abs(slopes(i)) * metric_norm / std::max(status.rNormsAbs(i), std::numeric_limits<RealScalar>::min());

            const bool finite_fit         = std::isfinite(slopes(i)) && std::isfinite(errors(i)) && std::isfinite(scaled);
            const bool significant_trend = std::abs(slopes(i)) > RealScalar{2} * errors(i);
            const bool significant_motion = scaled > cfg().ritz_stabilization_tolerance;
            const bool progressing         = finite_fit && significant_trend && significant_motion;
            if(progressing) return false;
            saturated = true;
        }
        return saturated;
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::refresh_direct_ritz_residuals() {
        Eigen::Index rows = std::min({cfg().nev, V.cols(), AV.cols(), BV.cols(), S.cols(), status.eigVal.size(), status.rNormsAbs.size()});
        if(rows <= 0) return;
        MatrixType av_direct = MultA(V.leftCols(rows));
        MatrixType bv_direct;
        if constexpr(form_ == grit::Form::GENERALIZED)
            bv_direct = MultB(V.leftCols(rows));
        else
            bv_direct = V.leftCols(rows);
        VectorReal rnorm_direct;
        MatrixType residual_direct = get_residuals(status.eigVal.topRows(rows), av_direct, bv_direct, rnorm_direct);
        AV.leftCols(rows)              = av_direct;
        BV.leftCols(rows)              = bv_direct;
        S.leftCols(rows)               = residual_direct;
        status.rNormsAbs.topRows(rows) = rnorm_direct;
    }

    template<typename Scalar, grit::Form form_>
    std::string base<Scalar, form_>::get_direct_ritz_diagnostics() {
        Eigen::Index rows = std::min({cfg().nev, V.cols(), AV.cols(), BV.cols(), status.eigVal.size(), status.rNormsAbs.size(), Eigen::Index{3}});
        if(rows <= 0) return {};
        MatrixType av_direct = MultA(V.leftCols(rows));
        MatrixType bv_direct;
        if constexpr(form_ == grit::Form::GENERALIZED)
            bv_direct = MultB(V.leftCols(rows));
        else
            bv_direct = V.leftCols(rows);
        VectorReal rnorm_direct;
        MatrixType residual_direct = get_residuals(status.eigVal.topRows(rows), av_direct, bv_direct, rnorm_direct);
        MatrixType residual_cached = AV.leftCols(rows) - BV.leftCols(rows) * status.eigVal.head(rows).asDiagonal();
        VectorReal eta_a            = (av_direct - AV.leftCols(rows)).colwise().norm().transpose();
        VectorReal eta              = (residual_direct - residual_cached).colwise().norm().transpose();
        VectorReal vnorm            = V.leftCols(rows).colwise().norm().transpose();
        VectorReal avnorm           = av_direct.colwise().norm().transpose();
        VectorReal bvnorm           = bv_direct.colwise().norm().transpose();
        VectorReal cancellation     = (avnorm.array() + status.eigVal.head(rows).cwiseAbs().array() * bvnorm.array()) /
                                  rnorm_direct.array().max(std::numeric_limits<RealScalar>::min());
        if constexpr(form_ == grit::Form::GENERALIZED) {
            VectorReal eta_b = (bv_direct - BV.leftCols(rows)).colwise().norm().transpose();
            VectorReal vbv(rows);
            for(Eigen::Index i = 0; i < rows; ++i) vbv(i) = std::real(V.col(i).dot(bv_direct.col(i)));
            return fmt::format("direct: eta={::.3e} etaA={::.3e} etaB={::.3e} |v|_l2={::.3e} vBv={::.3e} |Av|={::.3e} |Bv|={::.3e} "
                               "|r|={::.3e} cancel={::.3e}",
                               eta, eta_a, eta_b, vnorm, vbv, avnorm, bvnorm, rnorm_direct, cancellation);
        } else {
            return fmt::format("direct: eta={::.3e} etaA={::.3e} |v|_l2={::.3e} |Av|={::.3e} |r|={::.3e} cancel={::.3e}", eta, eta_a, vnorm,
                               avnorm, rnorm_direct, cancellation);
        }
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::update_condition_numbers() {
        status.condition_a = std::numeric_limits<RealScalar>::infinity();
        status.condition_b = form_ == grit::Form::STANDARD ? RealScalar{1} : std::numeric_limits<RealScalar>::infinity();
        if(Q.cols() == 0 || Q.rows() < Q.cols() || AQ.size() != Q.size()) return;

        Eigen::HouseholderQR<MatrixType> qr(Q);
        MatrixType R = qr.matrixQR().topLeftCorner(Q.cols(), Q.cols()).template triangularView<Eigen::Upper>();
        if((R.diagonal().array().abs() <= std::numeric_limits<RealScalar>::min()).any()) return;
        MatrixType R_inv = R.template triangularView<Eigen::Upper>().solve(MatrixType::Identity(R.rows(), R.cols()));
        MatrixType A_projected = R_inv.adjoint() * (Q.adjoint() * AQ) * R_inv;
        A_projected            = (A_projected + A_projected.adjoint()) * half;
        Eigen::SelfAdjointEigenSolver<MatrixType> es_a(A_projected, Eigen::EigenvaluesOnly);
        if(es_a.info() == Eigen::Success) {
            VectorReal values = es_a.eigenvalues().cwiseAbs();
            if(values.minCoeff() > std::numeric_limits<RealScalar>::min()) status.condition_a = values.maxCoeff() / values.minCoeff();
        }
        if constexpr(form_ == grit::Form::GENERALIZED) {
            if(BQ.size() != Q.size()) return;
            MatrixType B_projected = R_inv.adjoint() * (Q.adjoint() * BQ) * R_inv;
            B_projected            = (B_projected + B_projected.adjoint()) * half;
            Eigen::SelfAdjointEigenSolver<MatrixType> es_b(B_projected, Eigen::EigenvaluesOnly);
            if(es_b.info() == Eigen::Success) {
                VectorReal values = es_b.eigenvalues().cwiseAbs();
                if(values.minCoeff() > std::numeric_limits<RealScalar>::min()) status.condition_b = values.maxCoeff() / values.minCoeff();
            }
        }
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::updateStatus() {
        auto t_status_update = status.time_status_update.tic_token();

        // Eigenvalues are sorted in ascending order.
        status.oldVal  = status.eigVal.topRows(cfg().nev);
        status.eigVal  = T_evals(status.optIdx).topRows(cfg().nev);
        status.absDiff = (status.eigVal - status.oldVal).cwiseAbs();

        VectorReal denom = (RealScalar{0.5} * (status.eigVal + status.oldVal).array().abs()).matrix();
        denom            = denom.cwiseMax(VectorReal::Constant(denom.size(), std::numeric_limits<RealScalar>::min()));
        status.relDiff   = status.absDiff.cwiseQuotient(denom);

        VectorReal  targets             = rNormAbsTargets();
        bool        verify_convergence  = (status.rNormsAbs.topRows(cfg().nev).array() < targets.array()).all();
        if(verify_convergence) refresh_direct_ritz_residuals();

        // Accumulate counters after optional direct applications.
        status.num_matvecs_total                    += status.num_matvecs + status.num_matvecs_inner;
        status.num_matvecs_inner_total              += status.num_matvecs_inner;
        status.num_matvecs_a_total                  += status.num_matvecs_a + status.num_matvecs_a_inner;
        status.num_matvecs_b_total                  += status.num_matvecs_b + status.num_matvecs_b_inner;
        status.num_precond_total                    += status.num_precond + status.num_precond_inner;
        status.num_precond_inner_total              += status.num_precond_inner;
        status.num_operator_inner_total             += status.num_operator_inner;
        status.num_preconditioner_updates_total     += status.num_preconditioner_updates + status.num_preconditioner_updates_inner;
        status.num_preconditioner_apply_inner_total += status.num_preconditioner_apply_inner;
        status.num_preconditioner_apply_total       += status.num_precond + status.num_preconditioner_apply_inner;
        status.num_inner_iters_total                += status.num_inner_iters;

        status.op_norm_estimate = get_op_norm_estimate();

        status.rNormsAbsHistory.push_back(status.rNormsAbs.topRows(cfg().nev));
        status.eigVals_history.push_back(status.eigVal.topRows(cfg().nev));
        status.matvecs_history.push_back(status.num_matvecs + status.num_matvecs_inner);
        while(status.rNormsAbsHistory.size() > status.max_history_size) status.rNormsAbsHistory.pop_front();
        while(status.eigVals_history.size() > status.max_history_size) status.eigVals_history.pop_front();
        while(status.matvecs_history.size() > status.max_history_size) status.matvecs_history.pop_front();

        if(cfg().reltol > RealScalar{0}) {
            Eigen::Index rows = std::min({cfg().nev, status.rNormsAbs.size(), status.absDiff.size(), V.cols()});
            if constexpr(form_ == grit::Form::GENERALIZED) rows = std::min(rows, BV.cols());
            if(status.rnorm_abs_reference.size() != rows) status.rnorm_abs_reference = VectorReal::Zero(rows);
            VectorReal metric_norms(rows);
            for(Eigen::Index i = 0; i < rows; ++i) {
                if constexpr(form_ == grit::Form::GENERALIZED)
                    metric_norms(i) = BV.col(i).norm();
                else
                    metric_norms(i) = V.col(i).norm();
            }
            if(status.eigVals_history.size() >= 3) {
                VectorReal residual_scales = status.rNormsAbs.topRows(rows).cwiseMax(VectorReal::Constant(rows, std::numeric_limits<RealScalar>::min()));
                VectorReal stabilization =
                    get_standard_deviations(status.eigVals_history, false, -1).topRows(rows).cwiseProduct(metric_norms).cwiseQuotient(residual_scales);
                for(Eigen::Index i = 0; i < rows; ++i) {
                    if(stabilization(i) > cfg().ritz_stabilization_tolerance)
                        status.rnorm_abs_reference(i) = RealScalar{0};
                    else if(status.rnorm_abs_reference(i) <= RealScalar{0})
                        status.rnorm_abs_reference(i) = status.rNormsAbs(i);
                }
            } else
                status.rnorm_abs_reference.setZero();
        }

        if(eigVals_have_saturated())
            status.saturation_count_eigVal++;
        else
            status.saturation_count_eigVal = 0;

        if(rNorms_have_saturated())
            status.saturation_count_rNorm++;
        else
            status.saturation_count_rNorm = 0;

        constexpr auto beta      = RealScalar{0.5f};
        VectorReal     rNormsAbs = status.rNormsAbs.topRows(cfg().nev);
        targets                  = rNormAbsTargets();
        RealScalar     relGap    = status.gap * get_op_norm_estimate(status.eigVal.size() > 0 ? std::optional<RealScalar>{status.eigVal(0)} : std::nullopt);
        if(rNormsAbs.size() != targets.size()) throw std::logic_error("unequal residual norm and target sizes");
        status.residual_converged = (rNormsAbs.array() < targets.array()).all();
        status.residual_below_gap = rNormsAbs.maxCoeff() < beta * relGap;

        if(status.residual_converged) {
            std::string msg_rnorm_gap;
            if(std::isfinite(status.gap)) msg_rnorm_gap = fmt::format(" | gap {:.3e} (rel {:.3e})", status.gap, relGap);
            status.stopMessage.emplace_back(fmt::format("converged rNormAbs {::.3e} < target {::.3e}{} | outer_iter {} | mv {} | {:.3e} s", rNormsAbs, targets,
                                                        msg_rnorm_gap, status.outer_iter + 1, status.num_matvecs_total, status.time_elapsed.get_time()));
            status.stopReason |= StopReason::converged;
        }

        if(cfg().max_iters >= 0 && status.outer_iter + 1 >= cfg().max_iters) {
            status.stopMessage.emplace_back(fmt::format("outer iterations ({}) >= max_iters ({}) | mv {} | {:.3e} s", status.outer_iter + 1, cfg().max_iters,
                                                        status.num_matvecs_total, status.time_elapsed.get_time()));
            status.stopReason |= StopReason::max_iters;
        }

        if(cfg().max_matvecs >= 0 && status.num_matvecs_total >= cfg().max_matvecs) {
            status.stopMessage.emplace_back(fmt::format("num_matvecs_total ({}) >= max_matvecs ({}) | {:.3e} s", status.num_matvecs_total, cfg().max_matvecs,
                                                        status.time_elapsed.get_time()));
            status.stopReason |= StopReason::max_matvecs;
        }

        if(cfg().quit_when_saturated && std::min(status.saturation_count_eigVal, status.saturation_count_rNorm) >= status.saturation_count_max) {
            status.stopMessage.emplace_back(fmt::format("saturation_count (eigVal {} rNorm {}) >= saturation_count_max ({}) | outer_iter {} | mv {} | {:.3e} s",
                                                        status.saturation_count_eigVal, status.saturation_count_rNorm, status.saturation_count_max,
                                                        status.outer_iter + 1, status.num_matvecs_total, status.time_elapsed.get_time()));
            status.stopReason |= StopReason::ritz_value_stalled;
            status.stopReason |= StopReason::ritz_residual_stalled;
        }
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::printStatus() {
        if(!log || !log->should_log(spdlog::level::info) || status.eigVal.size() == 0 || status.rNormsAbs.size() == 0) return;

        std::string rCorrMsg;
        switch(residual_correction_type_internal) {
            case ResidualCorrectionType::NONE: rCorrMsg = "NO"; break;
            case ResidualCorrectionType::CHEAP_OLSEN: rCorrMsg = "CO"; break;
            case ResidualCorrectionType::FULL_OLSEN: rCorrMsg = "FO"; break;
            case ResidualCorrectionType::JACOBI_DAVIDSON: rCorrMsg = "JD"; break;
            case ResidualCorrectionType::AUTO: rCorrMsg = "AU"; break;
        }

        std::string innerMsg;
        if(status.num_matvecs_inner > 0 || status.num_operator_inner > 0 || status.num_precond_inner > 0) {
            const auto  time_solve_inner          = status.time_solve_inner.get_time_lap();
            const auto  time_operator_inner       = status.time_operator_inner.get_time_lap();
            const auto  time_preconditioner_inner = status.time_preconditioner_inner.get_time_lap();
            const auto  time_krylov_other         = std::max(0.0, time_solve_inner - time_operator_inner - time_preconditioner_inner);
            std::string pcMsg;
            if(status.num_precond_inner > 0) pcMsg = fmt::format(" pc={}/{:.1e}s", status.num_precond_inner, time_preconditioner_inner);
            innerMsg = fmt::format(" [{} it={} mv={} op={}/{:.1e}s{} err={:.2e} tol={:.2e} time={:.1e}s other={:.1e}s]", rCorrMsg, status.num_inner_iters,
                                   status.num_matvecs_inner, status.num_operator_inner, time_operator_inner, pcMsg, status.inner_error_last,
                                   status.inner_tol_last, time_solve_inner, time_krylov_other);
        }

        std::string timingMsg;
        if(status.outer_iter > 0) {
            timingMsg = fmt::format(" bld={:.1e}s corr={:.1e}s og={:.1e}s on={:.1e}s diag={:.1e}s extr={:.1e}s rst={:.1e}s stat={:.1e}s",
                                    status.time_build.get_time_lap(), status.time_residual_correction.get_time_lap(), status.time_orthogonalize.get_time_lap(),
                                    status.time_orthonormalize.get_time_lap(), status.time_diagonalize.get_time_lap(), status.time_extract_ritz.get_time_lap(),
                                    status.time_restart.get_time_lap(), status.time_status_update.get_time_lap());
        }

        RealScalar orthError = RealScalar{0};
        if(Q.cols() > 0) {
            MatrixType Gram = cfg().use_b_inner_product && BQ.size() == Q.size() ? Q.adjoint() * BQ : Q.adjoint() * Q;
            Gram            = (Gram + Gram.adjoint()).eval() / RealScalar{2};
            orthError       = (Gram - MatrixType::Identity(Gram.rows(), Gram.cols())).norm();
        }

        const auto  op_norm_estimate = get_op_norm_estimate(status.eigVal.size() > 0 ? std::optional<RealScalar>{status.eigVal(0)} : std::nullopt);
        const auto  rNormAbsTargets  = this->rNormAbsTargets();
        const auto  num_matvecs_iter = status.num_matvecs + status.num_matvecs_inner;
        const auto  num_precond_iter = status.num_precond + status.num_precond_inner;
        std::string pcMsg;
        if(num_precond_iter > 0) pcMsg = fmt::format(" pc={:>4}|{:<4}", num_precond_iter, status.num_precond_total);
        std::string rescaledMsg = cfg().use_rescaled_rnorm_tolerance ? " (rescaled)" : "";

        log->info("it={:>3} dim={} ritz={} mv={:>4}|{:<4}{} t={:.1e}|{:.1e}s{} eigVal={::.16f} orthErr={:.3e} "
                  "|rNorm|={::.3e} tgt={::.3e} ({:.2e}/mv) tol={:.1e} rtol={:.1e}{} sat={}:{}/{} col={} bs={} "
                  "|op|={:.2e} κ(A)={:.2e} κ(B)={:.2e}{}",
                  status.outer_iter, N, enum2sv(cfg().ritz), num_matvecs_iter, status.num_matvecs_total, pcMsg, status.time_elapsed.get_time_lap(),
                  status.time_elapsed.get_time(), innerMsg, status.eigVal, orthError, status.rNormsAbs, rNormAbsTargets, get_rNorms_log10_change_per_matvec(),
                  cfg().abstol, cfg().reltol, rescaledMsg, status.saturation_count_eigVal, status.saturation_count_rNorm, status.saturation_count_max, Q.cols(),
                  cfg().block_size, op_norm_estimate, status.condition_a, status.condition_b, timingMsg);

        if(status.outer_iter == status.num_outer_iters_last_restart && log->should_log(spdlog::level::debug))
            log->debug(get_direct_ritz_diagnostics());
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::printFinal() {
        if(!log || status.eigVal.size() == 0 || status.rNormsAbs.size() == 0) return;

        const Eigen::Index nev      = std::min(cfg().nev, std::min(status.eigVal.size(), status.rNormsAbs.size()));
        const Eigen::Index inner_mv = status.num_matvecs_inner_total;
        const Eigen::Index outer_mv = status.num_matvecs_total - inner_mv;
        const Eigen::Index inner_pc = status.num_precond_inner_total;
        const Eigen::Index outer_pc = status.num_precond_total - inner_pc;
        const RealScalar   inner_t  = status.time_solve_inner.get_time();
        const RealScalar   outer_t  = std::max<RealScalar>(RealScalar{0}, status.time_elapsed.get_time() - inner_t);

        for(Eigen::Index i = 0; i < nev; ++i) {
            const RealScalar op_norm_estimate = get_op_norm_estimate(std::optional<RealScalar>{status.eigVal(i)});
            std::string      outerPcMsg;
            std::string      innerPcMsg;
            std::string      totalPcMsg;
            std::string      stop_reason_msg = enum2s(status.stopReason);
            if(outer_pc > 0) outerPcMsg = fmt::format(" pc={}", outer_pc);
            if(inner_pc > 0) innerPcMsg = fmt::format(" pc={}", inner_pc);
            if(status.num_precond_total > 0) totalPcMsg = fmt::format(" pc={}", status.num_precond_total);
            if(has_flag(status.stopReason, StopReason::converged) && i < Eigen::Index{3}) {
                RealScalar abstol_target = cfg().abstol;
                if(cfg().use_rescaled_rnorm_tolerance) abstol_target *= rNormScale(i);
                const RealScalar reltol_target =
                    cfg().reltol > RealScalar{0} && i < status.rnorm_abs_reference.size() ? cfg().reltol * status.rnorm_abs_reference(i) : RealScalar{0};
                if(reltol_target > abstol_target)
                    stop_reason_msg += " (reltol)";
                else if(reltol_target == abstol_target && reltol_target > RealScalar{0})
                    stop_reason_msg += " (abstol+reltol)";
                else
                    stop_reason_msg += " (abstol)";
            }
            log->info("grit finished: {} | nev={}/{} eigVal={:.16e} |rNorm|={:.3e} | "
                      "outer: it={} mv={}{} t={:.1e}s | inner: it={} mv={} op={}{} t={:.1e}s | "
                      "total: mv={}{} t={:.1e}s | |op|={:.2e} κ(A)={:.2e} κ(B)={:.2e}",
                      stop_reason_msg, i + 1, nev, status.eigVal(i), status.rNormsAbs(i), status.outer_iter, outer_mv, outerPcMsg, outer_t,
                      status.num_inner_iters_total, inner_mv, status.num_operator_inner_total, innerPcMsg, inner_t, status.num_matvecs_total, totalPcMsg,
                      status.time_elapsed.get_time(), op_norm_estimate, status.condition_a, status.condition_b);
        }
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::restart_status_time_laps() {
        status.time_elapsed.restart_time_lap();
        status.time_solve_inner.restart_time_lap();
        status.time_matvecs.restart_time_lap();
        status.time_matvecs_a.restart_time_lap();
        status.time_matvecs_b.restart_time_lap();
        status.time_matvecs_a_inner.restart_time_lap();
        status.time_matvecs_b_inner.restart_time_lap();
        status.time_precond.restart_time_lap();
        status.time_preconditioner_inner.restart_time_lap();
        status.time_preconditioner_update.restart_time_lap();
        status.time_preconditioner_update_inner.restart_time_lap();
        status.time_preconditioner_apply_inner.restart_time_lap();
        status.time_operator_inner.restart_time_lap();
        status.time_project_left_inner.restart_time_lap();
        status.time_project_right_inner.restart_time_lap();
        status.time_residual_correction.restart_time_lap();
        status.time_build.restart_time_lap();
        status.time_orthogonalize.restart_time_lap();
        status.time_orthonormalize.restart_time_lap();
        status.time_orth_project.restart_time_lap();
        status.time_orth_factor.restart_time_lap();
        status.time_orth_update.restart_time_lap();
        status.time_orth_refresh.restart_time_lap();
        status.time_orth_mask.restart_time_lap();
        status.time_diagonalize.restart_time_lap();
        status.time_extract_ritz.restart_time_lap();
        status.time_restart.restart_time_lap();
        status.time_status_update.restart_time_lap();
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::run_user_callback() {
        return;
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::do_outer_iteration() {
        restart_status_time_laps();
        preamble();
        build();
        diagonalizeT();
        extractRitzVectors();
        updateStatus();
        printStatus();
        run_user_callback();
        status.outer_iter++;
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::run() {
        status.stopReason = StopReason::none;
        status.stopMessage.clear();
        status.residual_converged      = false;
        status.residual_below_gap      = false;
        status.saturation_count_eigVal = 0;
        status.saturation_count_rNorm  = 0;
        status.rNormsAbsHistory.clear();
        status.eigVals_history.clear();
        status.matvecs_history.clear();
        auto token_elapsed = status.time_elapsed.tic_token();
        if(status.outer_iter == 0) {
            init();
            printStatus();
            run_user_callback();
            status.outer_iter++;
        }
        while(true) {
            do_outer_iteration();
            if(status.stopReason != StopReason::none) break;
        }
        printFinal();
    }

}

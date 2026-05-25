#pragma once

#include <grit/form/base.h>
#include <grit/internal/precondition/JacobiDavidsonOperator.h>
#include <stdexcept>

namespace grit::form {
    template<typename Scalar>
    void base<Scalar>::OrthMeta::analyze_l2_orthonormality(const Eigen::Ref<const MatrixType> &Y) {
        if(Y.cols() == 0) return;
        MatrixType I = MatrixType::Identity(Y.cols(), Y.cols());
        Gram         = Y.adjoint() * Y;
        Gram_symm    = Gram;
        Gram_skew    = Gram;
        orthError    = (Gram - I).norm();
        symmError    = orthError;
        skewError    = orthError;
        Rdiag        = Gram_symm.diagonal().cwiseAbs().cwiseSqrt();
    }

    template<typename Scalar>
    void base<Scalar>::OrthMeta::analyze_b_orthonormality(const Eigen::Ref<const MatrixType> &Y, const Eigen::Ref<const MatrixType> &B_Y) {
        analyze_bm_orthonormality(Y, B_Y);
    }

    template<typename Scalar>
    void base<Scalar>::OrthMeta::analyze_bm_orthonormality(const Eigen::Ref<const MatrixType> &Y, const Eigen::Ref<const MatrixType> &B_Y) {
        if(Y.cols() != B_Y.cols() || Y.rows() != B_Y.rows()) return;
        MatrixType I  = MatrixType::Identity(Y.cols(), Y.cols());
        MatrixType G1 = Y.adjoint() * B_Y;
        MatrixType G2 = B_Y.adjoint() * Y;
        Gram          = G1;
        Gram_symm     = (G1 + G2) * half;
        Gram_skew     = (G1 - G2) * half;
        orthError     = (Gram - I).norm();
        symmError     = (Gram_symm - I).norm();
        skewError     = Gram_skew.norm();
        skewError_fwd = skewError;
        Rdiag         = Gram_symm.diagonal().cwiseAbs().cwiseSqrt();
    }

    template<typename Scalar>
    void base<Scalar>::OrthMeta::analyze_l2_orthogonality(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &Y) {
        if(Y.cols() == 0) return;
        Gram      = X.adjoint() * Y;
        Gram_symm = Gram;
        Gram_skew = Gram;
        orthError = Gram.norm();
        symmError = orthError;
        skewError = orthError;
        Rdiag     = Gram_symm.diagonal().cwiseAbs().cwiseSqrt();
    }

    template<typename Scalar>
    void base<Scalar>::OrthMeta::analyze_bm_orthogonality(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &B_X,
                                                          const Eigen::Ref<const MatrixType> &Y, const Eigen::Ref<const MatrixType> &B_Y) {
        if(Y.cols() != B_Y.cols() || Y.rows() != B_Y.rows()) return;
        if(X.cols() != B_X.cols() || X.rows() != B_X.rows()) return;
        if(Y.rows() != X.rows()) return;

        MatrixType G1 = X.adjoint() * B_Y;
        MatrixType G2 = B_X.adjoint() * Y;
        MatrixType I  = MatrixType::Identity(G1.rows(), G1.cols());

        Gram      = G1;
        Gram_symm = (G1 + G2) * half;
        Gram_skew = (G1 - G2) * half;

        orthError = (Gram - I).norm();
        symmError = (Gram_symm - I).norm();
        skewError = Gram_skew.norm();
        Rdiag     = Gram_symm.diagonal().cwiseAbs().cwiseSqrt();
    }

    template<typename Scalar>
    std::string_view base<Scalar>::ResidualCorrectionToString(ResidualCorrectionType rct) {
        switch(rct) {
            case ResidualCorrectionType::NONE: return "NONE";
            case ResidualCorrectionType::CHEAP_OLSEN: return "CHEAP_OLSEN";
            case ResidualCorrectionType::FULL_OLSEN: return "FULL_OLSEN";
            case ResidualCorrectionType::JACOBI_DAVIDSON: return "JACOBI_DAVIDSON";
            case ResidualCorrectionType::AUTO: return "AUTO";
        }
        return "NONE";
    }

    template<typename Scalar>
    typename base<Scalar>::ResidualCorrectionType base<Scalar>::StringToResidualCorrection(std::string_view rct) {
        if(rct == "NONE") return ResidualCorrectionType::NONE;
        if(rct == "CHEAP_OLSEN") return ResidualCorrectionType::CHEAP_OLSEN;
        if(rct == "FULL_OLSEN") return ResidualCorrectionType::FULL_OLSEN;
        if(rct == "JACOBI_DAVIDSON") return ResidualCorrectionType::JACOBI_DAVIDSON;
        if(rct == "AUTO") return ResidualCorrectionType::AUTO;
        return ResidualCorrectionType::NONE;
    }

    template<typename Scalar>
    void base<Scalar>::setLogger(spdlog::level::level_enum level, const std::string &name) {
        eiglog = Logger::getLogger(name.empty() ? "grit" : name);
        eiglog->set_level(level);
    }

    template<typename Scalar>
    base<Scalar>::base(Eigen::Index nev, Eigen::Index ncv, OptRitz ritz, const MatrixType &V, MatVec<Scalar> &A, spdlog::level::level_enum logLevel_)
        : logLevel(logLevel_), nev(nev), ncv(ncv), ritz(ritz), A(A), V(V) {
        setLogger(logLevel, "grit");
        N    = A.get_size();
        size = A.get_size();
        nev  = std::min(nev, N);
        ncv  = std::min(std::max(nev, ncv), N);
        b    = std::min(std::max(nev, b), std::max<Eigen::Index>(1, N / 2));
        status.rNorms.setOnes(nev);
        status.eigVal.setOnes(nev);
        status.oldVal.setOnes(nev);
        status.absDiff.setOnes(nev);
        status.relDiff.setOnes(nev);
    }

    template<typename Scalar>
    auto base<Scalar>::get_residuals(const Eigen::Ref<VectorReal> &Y, const Eigen::Ref<MatrixType> &AV, const Eigen::Ref<MatrixType> &BV)
        -> std::pair<MatrixType, VectorReal> {
        MatrixType S        = AV - BV * Y.asDiagonal();
        VectorReal N        = S.colwise().norm().transpose();
        VectorReal AV_norms = AV.colwise().norm().transpose();
        VectorReal BV_norms = BV.colwise().norm().transpose();
        VectorReal D        = AV_norms.array() + BV_norms.array() * Y.cwiseAbs().array();
        D                   = D.cwiseMax(VectorReal::Constant(D.size(), std::numeric_limits<RealScalar>::min()));
        VectorReal R        = N.cwiseQuotient(D);
        return {S, R};
    }

    template<typename Scalar>
    typename base<Scalar>::RealScalar base<Scalar>::rNormTol([[maybe_unused]] Eigen::Index n) const {
        auto tol = abstol;
        if(reltol > RealScalar{0} && n < status.rNorms_init.size()) tol = std::clamp(reltol * status.rNorms_init(n), tol, RealScalar{0.99f});
        return tol;
    }

    template<typename Scalar>
    typename base<Scalar>::VectorReal base<Scalar>::rNormTols() const {
        VectorReal rNormTols(nev);
        for(Eigen::Index n = 0; n < nev; ++n) rNormTols(n) = rNormTol(n);
        return rNormTols;
    }

    template<typename Scalar>
    typename base<Scalar>::RealScalar base<Scalar>::get_rNorms_log10_change_per_matvec() {
        if(status.rNorms_history.size() < 2ul) return RealScalar{0};
        auto size = status.rNorms_history.size();
        assert(size == status.matvecs_history.size());

        auto rNorm_change = status.rNorms_history[size - 1].array() / status.rNorms_history[size - 2].array();
        auto sum_matvecs  = status.matvecs_history[size - 1] + status.matvecs_history[size - 2];
        if(sum_matvecs <= 0) return RealScalar{0};
        return std::log10(rNorm_change.minCoeff()) / static_cast<RealScalar>(sum_matvecs);
    }

    template<typename Scalar>
    typename base<Scalar>::RealScalar base<Scalar>::get_op_norm_estimate(std::optional<RealScalar> eigval) const {
        auto op_norm_estimate = std::max(RealScalar{1}, status.op_norm_estimate);
        if(!std::isfinite(op_norm_estimate)) op_norm_estimate = RealScalar{1};

        if(Q.size() == 0 || Q.norm() == RealScalar{0}) return std::max(op_norm_estimate, A.get_op_norm());

        if(is_generalized_problem()) {
            auto A_maxeval = T_evals.size() > 0 ? T_evals.cwiseAbs().maxCoeff() : RealScalar{1};
            auto A_maxnorm = AQ.size() == Q.size() ? AQ.norm() / Q.norm() : RealScalar{1};
            auto B_maxnorm = BQ.size() == Q.size() ? BQ.norm() / Q.norm() : RealScalar{1};
            auto abs_lambda = std::abs(eigval.value_or(status.eigVal.size() > 0 ? status.eigVal(0) : RealScalar{1}));
            return std::max({op_norm_estimate, A_maxeval, A_maxnorm + abs_lambda * B_maxnorm, A.get_op_norm()});
        }

        auto A_maxeval = T_evals.size() > 0 ? T_evals.cwiseAbs().maxCoeff() : RealScalar{1};
        auto A_maxnorm = AQ.size() == Q.size() ? AQ.norm() / Q.norm() : RealScalar{1};
        return std::max({op_norm_estimate, A_maxeval, A_maxnorm, A.get_op_norm()});
    }

    template<typename Scalar>
    typename base<Scalar>::MatrixType base<Scalar>::MultA(const Eigen::Ref<const MatrixType> &X) {
        auto token_matvecs  = status.time_matvecs.tic_token();
        status.num_matvecs += X.cols();
        return A.MultAX(X);
    }

    template<typename Scalar>
    typename base<Scalar>::MatrixType base<Scalar>::MultP(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const VectorReal> &evals) {
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

    template<typename Scalar>
    void base<Scalar>::adjust_residual_correction_type() {
        residual_correction_type_internal = residual_correction_type;
        if(residual_correction_type_internal == ResidualCorrectionType::AUTO) {
            residual_correction_type_internal = ResidualCorrectionType::NONE;
            if(inner_tol < RealScalar{1e-1f} || status.num_matvecs_inner > 300) residual_correction_type_internal = ResidualCorrectionType::CHEAP_OLSEN;
            if(inner_tol < RealScalar{1e-3f} || status.num_matvecs_inner > 1000) residual_correction_type_internal = ResidualCorrectionType::FULL_OLSEN;
            if(inner_tol < RealScalar{1e-5f} || status.num_matvecs_inner > 2000) residual_correction_type_internal = ResidualCorrectionType::JACOBI_DAVIDSON;
        }
    }

    template<typename Scalar>
    typename base<Scalar>::MatrixType base<Scalar>::cheap_Olsen_correction(const MatrixType &V, const MatrixType &S) {
        MatrixType D(S.rows(), S.cols());

        assert(V.allFinite());
        assert(S.allFinite());
        for(long i = 0; i < S.cols(); ++i) {
            auto d           = D.col(i);
            auto v           = V.col(i);
            auto s           = S.col(i);
            auto numerator   = Scalar{1};
            auto denominator = Scalar{1};

            if(use_b_inner_product && BV.rows() == V.rows() && BV.cols() > i) {
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

    template<typename Scalar>
    typename base<Scalar>::MatrixType base<Scalar>::full_Olsen_correction(const MatrixType &V, const MatrixType &S) {
        MatrixType MV;
        MatrixType MS;
        MatrixType coeffs;
        auto       Y = T_evals(status.optIdx);

        if(use_b_inner_product && BV.rows() == V.rows() && BV.cols() == V.cols()) {
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

    template<typename Scalar>
    typename base<Scalar>::MatrixType base<Scalar>::jacobi_davidson_l2_correction(const MatrixType &V, const MatrixType &S, const VectorReal &evals) {
        assert(V.rows() == S.rows());
        assert(V.cols() == S.cols());
        assert(!use_b_inner_product);

        auto MatrixOp = [this](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
            auto token_matvecs_inner  = status.time_matvecs_inner.tic_token();
            status.num_matvecs_inner += X.cols();
            if(is_generalized_problem()) return MultB_inner(X);
            return A.MultAX(X);
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
                cfg.maxiters             = inner_iters;
                cfg.tolerance            = inner_tol;
                cfg.theta                = th;
                cfg.preconditioner_apply = [this](const Eigen::Ref<const VectorType> &x, Eigen::Ref<VectorType> y, RealScalar theta) -> void {
                    A.preconditioner_apply(x, y, theta);
                };

                auto ResidualOp = [this, th](const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> HX) -> void {
                    auto token_matvecs_inner = status.time_matvecs_inner.tic_token();
                    HX.resize(X.rows(), X.cols());
                    if(use_jd_b_only) {
                        HX.noalias()              = MultB_inner(X);
                        status.num_matvecs_inner += X.cols();
                    } else if(is_generalized_problem()) {
                        HX.noalias()              = A.MultAX(X) - th * MultB_inner(X);
                        status.num_matvecs_inner += 2 * X.cols();
                    } else {
                        HX.noalias()              = A.MultAX(X) - th * X;
                        status.num_matvecs_inner += X.cols();
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
        status.num_precond += b;
        return D;
    }

    template<typename Scalar>
    typename base<Scalar>::MatrixType base<Scalar>::jacobi_davidson_bm_correction(const MatrixType &V, const MatrixType &BV, const MatrixType &S,
                                                                                  const VectorReal &evals) {
        assert(is_generalized_problem());
        assert(use_b_inner_product);
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
            if(i >= nev) {
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
                cfg.maxiters             = inner_iters;
                cfg.tolerance            = inner_tol;
                cfg.theta                = th;
                cfg.preconditioner_apply = [this](const Eigen::Ref<const VectorType> &x, Eigen::Ref<VectorType> y, RealScalar theta) -> void {
                    A.preconditioner_apply(x, y, theta);
                };

                auto MatrixOp = [this](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
                    auto token_matvecs_inner  = status.time_matvecs_inner.tic_token();
                    status.num_matvecs_inner += X.cols();
                    return MultB_inner(X);
                };

                auto ResidualOp = [this, th](const Eigen::Ref<const MatrixType> &X, Eigen::Ref<MatrixType> HX) -> void {
                    auto token_matvecs_inner = status.time_matvecs_inner.tic_token();
                    HX.resize(X.rows(), X.cols());
                    if(use_jd_b_only) {
                        HX.noalias()              = MultB_inner(X);
                        status.num_matvecs_inner += X.cols();
                    } else {
                        HX.noalias()              = A.MultAX(X) - th * MultB_inner(X);
                        status.num_matvecs_inner += 2 * X.cols();
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
        status.num_precond += b;
        return D;
    }

    template<typename Scalar>
    typename base<Scalar>::MatrixType base<Scalar>::get_sBlock(const MatrixType &S_in) {
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
                if(use_jd_b_only && !is_generalized_problem()) throw std::runtime_error("use_jd_b_only requires a generalized eigenvalue problem");
                if(use_b_inner_product) {
                    S.noalias() = jacobi_davidson_bm_correction(V, BV, S, Y);
                } else {
                    S.noalias() = jacobi_davidson_l2_correction(V, S, Y);
                }
                break;
        }
        assert_allFinite(S);
        return S;
    }

    template<typename Scalar>
    std::vector<Eigen::Index> base<Scalar>::get_ritz_indices(OptRitz ritz, Eigen::Index offset, Eigen::Index num, const VectorReal &evals) const {
        std::vector<Eigen::Index> indices;
        switch(ritz) {
            case OptRitz::SR: indices = getIndices(evals, offset, num, std::less<RealScalar>()); break;
            case OptRitz::LR: indices = getIndices(evals, offset, num, std::greater<RealScalar>()); break;
            case OptRitz::SM: {
                VectorReal abs_evals = evals.cwiseAbs();
                indices              = getIndices(abs_evals, offset, num, std::less<RealScalar>());
                break;
            }
            case OptRitz::LM: {
                VectorReal abs_evals = evals.cwiseAbs();
                indices              = getIndices(abs_evals, offset, num, std::greater<RealScalar>());
                break;
            }
            case OptRitz::NONE: indices = getIndices(evals, offset, num, std::less<RealScalar>()); break;
        }
        return indices;
    }

    template<typename Scalar>
    void base<Scalar>::init() {
        assert(N == A.get_size() && "A must have same dimension");
        nev                         = std::min(nev, N);
        ncv                         = std::min(std::max(nev, ncv), N);
        b                           = std::min(std::max(nev, b), N / 2);
        status.saturation_count_max = ncv;
        Eigen::ColPivHouseholderQR<MatrixType> cpqr;

        // Step 0: Construct and orthonormalize the initial block V.
        // We aim to construct V = [v[0]...v[b-1]], where v are ritz eigenvectors.
        // If V has fewer than b columns, we pad it with random vectors and orthonormalize with ColPivHouseholderQR.
        // If V has more than b columns, we discard the overshooting columns after QR.
        // If after QR we have fewer than b columns, we pad again (this is a very unlikely event)
        assert(V.size() == 0 or N == V.rows());
        for(long i = 0; i < 2; ++i) {
            if(V.cols() < b) {
                // Pad with random vectors
                auto vc = V.cols();
                V.conservativeResize(N, b);
                auto Vrc = V.rightCols(b - vc);
                for(auto vj : Vrc.colwise()) { vj = Eigen::VectorXf::Random(vj.size()).template cast<Scalar>(); }
            }
            // Orthonormalize V.
            // Discard columns if there are more than b (this is not expected, but also not an error)
            cpqr.compute(V);
            auto rank = std::min<Eigen::Index>(cpqr.rank(), b);
            V         = cpqr.householderQ().setLength(rank) * MatrixType::Identity(N, rank);
            if(V.cols() == b) break;
        }

        auto block_orthonormalize = [&] {
            auto m       = OrthMeta();
            m.maskPolicy = MaskPolicy::COMPRESS;
            if(is_generalized_problem()) {
                if(use_b_inner_product) {
                    block_bm_orthonormalize(V, AV, BV, m);
                } else {
                    block_l2_orthonormalize(V, AV, BV, m);
                }
            } else {
                block_l2_orthonormalize(V, AV, m);
            }
        };

        assert(V.cols() == b);
        if(status.iter == 0) {
            // Make sure we start with ritz vectors in V, so that the first Lanczos loop produces proper residuals.
            if(is_generalized_problem()) {
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
                status.optIdx = get_ritz_indices(ritz, 0, b, T_evals);
                MatrixType Z  = T_evecs(Eigen::placeholders::all, status.optIdx);
                VectorReal Y  = T_evals(status.optIdx);
                V             = Q * Z;  // Now V has b columns mixed according to the selected columns in T_evecs
                AV            = AQ * Z; // Now AV has b columns mixed according to the selected columns in T_evecs
                BV            = BQ * Z; // Now BV has b columns mixed according to the selected columns in T_evecs

                RealScalar min_sep =
                    T_evals.size() <= 1 ? RealScalar{1} : (T_evals.tail(T_evals.size() - 1) - T_evals.head(T_evals.size() - 1)).cwiseAbs().minCoeff();
                auto select1       = get_ritz_indices(ritz, 0, 1, T_evals);
                auto A_max_abs     = std::max({T1.cwiseAbs().maxCoeff(), AV.norm() / V.norm(), RealScalar{1}});
                auto B_max_abs     = std::max({T2.cwiseAbs().maxCoeff(), BV.norm() / V.norm(), RealScalar{1}});
                status.sensitivity = (A_max_abs + T_evals(select1).cwiseAbs().coeff(0) * B_max_abs) / min_sep;

                auto AB_max_abs         = T_evals.cwiseAbs().maxCoeff();
                auto AB_min_abs         = T_evals.cwiseAbs().minCoeff();
                status.condition        = AB_max_abs / std::max(AB_min_abs, std::numeric_limits<RealScalar>::min());
                status.op_norm_estimate = A_max_abs + T_evals(select1).cwiseAbs().coeff(0) * B_max_abs;
                // We may need to orthonormalize V in generalized problems
                block_orthonormalize();

                std::tie(S, status.rNorms) = get_residuals(Y, AV, BV);
                status.eigVal              = Y.topRows(nev); // Make sure we only take nev values here. In general, nev <= b
            } else {
                block_orthonormalize();
                Q  = V;
                AQ = MultA(V);
                T  = Q.adjoint() * AQ;
                T  = RealScalar{0.5f} * (T.adjoint() + T); // Symmetrize
                Eigen::SelfAdjointEigenSolver<MatrixType> es(T);
                T_evecs                    = es.eigenvectors();
                T_evals                    = es.eigenvalues();
                status.optIdx              = get_ritz_indices(ritz, 0, b, T_evals);
                MatrixType Z               = T_evecs(Eigen::placeholders::all, status.optIdx);
                VectorReal Y               = T_evals(status.optIdx);
                V                          = Q * Z; // Now V has b columns mixed according to the selected columns in T_evecs
                AV                         = AQ * Z;
                BV                         = V;
                BQ                         = Q;
                std::tie(S, status.rNorms) = get_residuals(Y, AV, V);
                status.eigVal              = Y.topRows(nev); // Make sure we only take nev values here. In general, nev <= b

                auto A_max_abs          = T_evals.cwiseAbs().maxCoeff();
                auto A_min_abs          = T_evals.cwiseAbs().minCoeff();
                status.condition        = A_max_abs / std::max(A_min_abs, std::numeric_limits<RealScalar>::min());
                status.op_norm_estimate = std::max({A_max_abs, AQ.norm() / Q.norm(), RealScalar{1}});
            }
        }
        status.rNorms_init = status.rNorms;
        assert(V.cols() == b);
        assert_allFinite(V);
        last_log_time.tic();
    }

    template<typename Scalar>
    void base<Scalar>::extractRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &AV, MatrixType &S, VectorReal &rNorms) {
        MatrixType Z = T_evecs(Eigen::placeholders::all, optIdx);
        VectorReal Y = T_evals(optIdx);

        V                   = Q * Z;
        AV                  = AQ * Z;
        std::tie(S, rNorms) = get_residuals(Y, AV, V);
    }

    template<typename Scalar>
    void base<Scalar>::extractRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &AV, MatrixType &BV, MatrixType &S,
                                          VectorReal &rNorms) {
        MatrixType Z = T_evecs(Eigen::placeholders::all, optIdx);
        VectorReal Y = T_evals(optIdx);

        V.noalias()         = Q * Z;
        AV.noalias()        = AQ * Z;
        BV.noalias()        = BQ * Z;
        std::tie(S, rNorms) = get_residuals(Y, AV, BV);
    }

    template<typename Scalar>
    void base<Scalar>::extractRitzVectors() {
        if(status.stopReason != StopReason::none) return;
        if(T_evals.size() < b) return;

        Eigen::Index k     = std::min(maxPrevBlocks * b, T_evals.size());
        Eigen::Index nritz = std::max({nev, b, k});

        status.optIdx = get_ritz_indices(ritz, 0, nritz, T_evals);

        if(use_refined_rayleigh_ritz) {
            if(is_generalized_problem()) {
                refinedRitzVectors(status.optIdx, V, AV, BV, S, status.rNorms);
            } else {
                refinedRitzVectors(status.optIdx, V, AV, S, status.rNorms);
                BV = V;
            }
        } else {
            if(is_generalized_problem()) {
                extractRitzVectors(status.optIdx, V, AV, BV, S, status.rNorms);
            } else {
                extractRitzVectors(status.optIdx, V, AV, S, status.rNorms);
                BV = V;
            }
        }

        K_prev = K;
        K      = V.leftCols(k);

        if(k > b) {
            V.conservativeResize(Eigen::NoChange, b);
            AV.conservativeResize(Eigen::NoChange, b);
            BV.conservativeResize(Eigen::NoChange, b);
            S.conservativeResize(Eigen::NoChange, b);
            status.rNorms.conservativeResize(b);
        }

        if(status.rNorms_init.size() != status.rNorms.size()) status.rNorms_init = status.rNorms;
    }

    template<typename Scalar>
    void base<Scalar>::orthonormalize_Z(Eigen::Ref<MatrixType> Z, const Eigen::Ref<const MatrixType> &T2) {
        if(!use_b_inner_product) {
            hhqr.compute(Z);
            Z = hhqr.householderQ() * MatrixType::Identity(Z.rows(), Z.cols());
        } else {
            MatrixType G    = Z.adjoint() * T2 * Z;
            G               = (G + G.adjoint()).eval() * half;
            auto       es   = Eigen::SelfAdjointEigenSolver<MatrixType>(G);
            VectorReal D    = es.eigenvalues();
            MatrixType U    = es.eigenvectors();
            RealScalar cut  = 100 * eps * static_cast<RealScalar>(D.size()) * D.cwiseAbs().maxCoeff();
            RealScalar cut2 = cut * cut;
            for(Eigen::Index j = 0; j < D.size(); ++j) {
                if(D(j) < cut2) { D(j) = std::max(D(j), cut2); }
            }
            Z *= U * D.cwiseInverse().cwiseSqrt().asDiagonal() * U.adjoint();
        }
    }

    template<typename Scalar>
    typename base<Scalar>::MatrixType base<Scalar>::get_refined_ritz_eigenvectors_std(const Eigen::Ref<const MatrixType> &Z,
                                                                                      const Eigen::Ref<const VectorReal> &Y, const MatrixType &Q,
                                                                                      const MatrixType &AQ) {
        assert(!use_b_inner_product);
        assert(Z.cols() == Y.size());
        Eigen::JacobiSVD<MatrixType, Eigen::ComputeThinV> svd;
        MatrixType                                        Z_ref(Z.rows(), Z.cols());
        MatrixType                                        T2Z_ref = MatrixType::Zero(Z.rows(), Z.cols());
        for(Eigen::Index j = 0; j < Y.size(); ++j) {
            const auto &theta = Y(j);
            MatrixType  M     = AQ - theta * Q;
            svd.compute(M);

            Eigen::Index min_idx;
            svd.singularValues().minCoeff(&min_idx);

            if(svd.info() == Eigen::Success) {
                Z_ref.col(j) = svd.matrixV().col(min_idx);
            } else {
                Z_ref.col(j)            = Z.col(j);
                RealScalar refinedRnorm = svd.singularValues()(min_idx);
                if(eiglog) eiglog->warn("refinement failed on ritz vector {} | refined rnorm={:.5e} | info {} ", j, refinedRnorm, static_cast<int>(svd.info()));
            }
        }
        return Z_ref;
    }

    template<typename Scalar>
    typename base<Scalar>::MatrixType base<Scalar>::get_refined_ritz_eigenvectors_gen(const Eigen::Ref<const MatrixType> &Z,
                                                                                      const Eigen::Ref<const VectorReal> &Y, const MatrixType &AQ,
                                                                                      const MatrixType &BQ) {
        assert(is_generalized_problem());
        assert(Z.cols() == Y.size());
        Eigen::JacobiSVD<MatrixType, Eigen::ComputeThinV> svd;
        MatrixType                                        Z_ref(Z.rows(), Z.cols());
        MatrixType                                        T2Z_ref = MatrixType::Zero(Z.rows(), Z.cols());
        for(Eigen::Index j = 0; j < Y.size(); ++j) {
            const auto &theta = Y(j);
            MatrixType  M     = AQ - theta * BQ;

            svd.compute(M);

            if(svd.info() == Eigen::Success) {
                Eigen::Index min_idx;
                svd.singularValues().minCoeff(&min_idx);

                auto zj   = Z_ref.col(j);
                auto t2zj = T2Z_ref.col(j);
                zj        = svd.matrixV().col(min_idx);

                if(use_b_inner_product) {
                    t2zj = T2 * zj;
                } else {
                    t2zj = zj;
                }

                if(j > 0) {
                    auto Z_prev   = Z_ref.leftCols(j);
                    auto T2Z_prev = T2Z_ref.leftCols(j);

                    MatrixType Gxx = Z_prev.adjoint() * T2Z_prev;
                    Gxx            = (Gxx + Gxx.adjoint()).eval() * half;

                    MatrixType Gxy = Z_prev.adjoint() * t2zj;
                    MatrixType W   = Gxx.ldlt().solve(Gxy);

                    zj.noalias()   -= Z_prev * W;
                    t2zj.noalias() -= T2Z_prev * W;
                }

                RealScalar norm = std::sqrt(std::max<RealScalar>(0, std::real(zj.dot(t2zj))));
                if(norm < normTol) {
                    zj.setZero();
                    t2zj.setZero();
                    continue;
                }
                zj   /= norm;
                t2zj /= norm;

            } else {
                Z_ref.col(j) = Z.col(j);
                if(eiglog) eiglog->warn("refinement failed on ritz vector {} | info {} ", j, static_cast<int>(svd.info()));
            }
        }
        return Z_ref;
    }

    template<typename Scalar>
    std::pair<typename base<Scalar>::MatrixType, typename base<Scalar>::MatrixType>
        base<Scalar>::get_bm_normalizer_for_the_projected_pencil(const MatrixType &T2) {
        MatrixType T2h = (T2 + T2.adjoint()) / RealScalar{2};

        auto es = Eigen::SelfAdjointEigenSolver<MatrixType>(T2h, Eigen::ComputeEigenvectors);
        if(es.info() != Eigen::Success) throw std::runtime_error("get_bm_normalizer_for_the_projected_pencil: eigensolver failed");

        auto U = es.eigenvectors();
        auto D = es.eigenvalues();

        const RealScalar Dmax = std::max<RealScalar>(RealScalar{1}, D.cwiseAbs().maxCoeff());
        const RealScalar tau  = RealScalar{10} * eps * Dmax;

        if(D.minCoeff() <= RealScalar{0} && eiglog) eiglog->warn("Projected T2 is numerically indefinite: min eval {:.3e}", D.minCoeff());

        for(Eigen::Index k = 0; k < D.size(); ++k) D(k) = std::max(D(k), tau);

        return {U * D.cwiseInverse().cwiseSqrt().asDiagonal() * U.adjoint(), U * D.cwiseSqrt().asDiagonal() * U.adjoint()};
    }

    template<typename Scalar>
    typename base<Scalar>::MatrixType base<Scalar>::get_optimal_rayleigh_ritz_matrix(const MatrixType &Z_rr, const MatrixType &Z_ref, const MatrixType &T1,
                                                                                     const MatrixType &T2) {
        assert(Z_rr.size() > 0);
        assert(Z_rr.rows() == Z_ref.rows());
        assert(Z_rr.cols() == Z_ref.cols());
        assert(Z_rr.rows() == T1.rows());
        assert(Z_rr.rows() == T2.rows());
        MatrixType Z(Z_rr.rows(), Z_rr.cols());

        MatrixType T1h = (T1.adjoint() + T1) * half;
        MatrixType T2h = (T2.adjoint() + T2) * half;

        MatrixType I = MatrixType::Identity(2, 2);
        for(Eigen::Index k = 0; k < Z.cols(); ++k) {
            using M2Type = Eigen::Matrix<Scalar, 2, 2>;
            M2Type     A2(2, 2), B2(2, 2);
            VectorType z0 = Z_rr.col(k);
            VectorType z1 = Z_ref.col(k);
            A2(0, 0)      = z0.adjoint() * T1h * z0;
            A2(1, 0)      = z1.adjoint() * T1h * z0;
            A2(0, 1)      = z0.adjoint() * T1h * z1;
            A2(1, 1)      = z1.adjoint() * T1h * z1;

            B2(0, 0) = z0.adjoint() * T2h * z0;
            B2(1, 0) = z1.adjoint() * T2h * z0;
            B2(0, 1) = z0.adjoint() * T2h * z1;
            B2(1, 1) = z1.adjoint() * T2h * z1;

            RealScalar tau  = 10 * eps * std::max(RealScalar{1}, std::real(B2.trace()) * half);
            B2             += I * tau;

            A2 = (A2.adjoint() + A2) * half;
            B2 = (B2.adjoint() + B2) * half;

            auto ges = Eigen::GeneralizedSelfAdjointEigenSolver<M2Type>(A2, B2, Eigen::Ax_lBx);
            if(ges.info() == Eigen::Success) {
                auto select1 = get_ritz_indices(ritz, 0, 1, ges.eigenvalues());
                auto v       = ges.eigenvectors().col(select1.at(0));
                Z.col(k)     = z0 * v(0) + z1 * v(1);
            } else {
                if(eiglog) eiglog->warn("generalized refined Rayleigh-Ritz 2x2 solve failed");
                Z.col(k) = z0;
            }
        }

        orthonormalize_Z(Z, T2h);

        return Z;
    }

    template<typename Scalar>
    void base<Scalar>::refinedRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &AQ, MatrixType &BQ, MatrixType &S,
                                          VectorReal &rNorms) {
        assert(is_generalized_problem());
        VectorReal Y     = T_evals(optIdx);
        MatrixType Z_rr  = T_evecs(Eigen::placeholders::all, optIdx);
        MatrixType Z_ref = get_refined_ritz_eigenvectors_gen(Z_rr, Y, AQ, BQ);
        MatrixType Z_opt = get_optimal_rayleigh_ritz_matrix(Z_rr, Z_ref, T1, T2);

        V.noalias()  = Q * Z_opt;
        AQ.noalias() = this->AQ * Z_opt;
        BQ.noalias() = this->BQ * Z_opt;

        if(use_rayleigh_quotients_instead_of_evals) {
            VectorReal rq1  = (V.adjoint() * AQ).diagonal().real();
            VectorReal rq2  = (V.adjoint() * BQ).diagonal().real();
            T_evals(optIdx) = rq1.cwiseQuotient(rq2);
            Y               = T_evals(optIdx);
        }

        std::tie(S, rNorms) = get_residuals(Y, AQ, BQ);
    }

    template<typename Scalar>
    void base<Scalar>::refinedRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &AQ, MatrixType &S, VectorReal &rNorms) {
        MatrixType Z     = T_evecs(Eigen::placeholders::all, optIdx);
        VectorReal Y     = T_evals(optIdx);
        MatrixType Z_ref = get_refined_ritz_eigenvectors_std(Z, Y, Q, this->AQ);

        V  = Q * Z_ref;
        AQ = this->AQ * Z_ref;

        if(use_rayleigh_quotients_instead_of_evals) {
            Y = (V.adjoint() * AQ).diagonal().real();
        }

        std::tie(S, rNorms) = get_residuals(Y, AQ, V);
    }

    template<typename Scalar>
    void base<Scalar>::refinedRitzVectors() {
        if(!use_refined_rayleigh_ritz) return;
        if(status.rNorms.size() == 0) throw std::runtime_error("refineRitzVectors() called before extractRitzVectors()");
        if(is_generalized_problem()) {
            refinedRitzVectors(status.optIdx, V, AV, BV, S, status.rNorms);
        } else {
            refinedRitzVectors(status.optIdx, V, AV, S, status.rNorms);
        }
    }

    template<typename Scalar>
    void base<Scalar>::preamble() {
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

        adjust_inner_tolerance(S);
        adjust_residual_correction_type();
    }

    template<typename Scalar>
    void base<Scalar>::adjust_inner_tolerance([[maybe_unused]] const Eigen::Ref<const MatrixType> &S) {
        if(!use_adaptive_inner_tolerance) return;
        if(status.num_iters_inner_prev == 0) return;
        // auto Snorm = S.leftCols(nev).colwise().norm().minCoeff();

        auto set_cfg = [&]() {
            auto oldtol = std::max(eps, inner_tol);
            auto oldits = status.num_iters_inner_prev;

            inner_tol = oldtol; // std::min<RealScalar>({oldtol, std::sqrt(Snorm)});
            if(status.iter > 0) {
                if(oldits < 100l) inner_tol *= std::sqrt(half);
                if(oldits > inner_iters / 2) inner_tol *= std::sqrt(RealScalar{2});
            }

            inner_tol = std::clamp(inner_tol, eps, RealScalar{0.75f});
            // RealScalar maxiters = RealScalar{50l} / inner_tol;
            // inner_iters = std::clamp(safe_cast<long>(maxiters), 50l, 200l);

            // RealScalar tol_rnorm = std::pow(Snorm, RealScalar{0.382f});
            // RealScalar tol_rnorm = RealScalar{1e-4f}; // std::pow(Snorm, RealScalar{0.5f});
            // RealScalar tol_old   = inner_tol;
            // RealScalar tol_min = RealScalar{0.1f}; // std::sqrt(eps);
            // RealScalar tol_max = RealScalar{0.1f};
            // RealScalar tol_min = RealScalar{1e-20f}; // std::sqrt(eps);
            // RealScalar tol_max = RealScalar{1e-1f};

            // if(status.iter > 0) {
            //     if(oldits < 50) inner_tol = std::min(inner_tol, oldtol * half);
            //     if(oldits > inner_iters / 2) inner_tol *= RealScalar{2};
            // }
            // inner_tol = std::clamp(tol_rnorm, tol_min, tol_max);

            // inner_tol = RealScalar{1e-2f};
            if(inner_tol != oldtol and eiglog and oldits > 0) eiglog->info("tol {:.2e} -> {:.2e} oldits {}", oldtol, inner_tol, oldits);
        };

        set_cfg();
    }

    template<typename Scalar>
    typename base<Scalar>::VectorReal base<Scalar>::get_standard_deviations(const std::deque<VectorReal> &v, bool apply_log10) {
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

    template<typename Scalar>
    typename base<Scalar>::VectorReal base<Scalar>::get_slopes(const std::deque<VectorReal> &v, bool apply_log10) {
        auto get_slope = [](const VectorReal &x, const VectorReal &y) -> RealScalar {
            assert(x.size() == y.size());
            assert(x.size() >= 2);
            auto xmean = x.mean();
            auto ymean = y.mean();
            auto sxy   = (x.array() - xmean).matrix().dot((y.array() - ymean).matrix());
            auto sxx   = (x.array() - xmean).matrix().dot((x.array() - xmean).matrix());
            return sxy / sxx;
        };
        if(v.empty()) return {};

        auto       m = static_cast<Eigen::Index>(v.size());
        auto       n = static_cast<Eigen::Index>(v.front().size());
        VectorReal x = VectorReal::LinSpaced(m, RealScalar(0), RealScalar(m - 1));
        VectorReal slopes(n);
        using history_size_type = typename std::deque<VectorReal>::size_type;
        for(Eigen::Index j = 0; j < n; ++j) {
            VectorReal y(m);
            for(Eigen::Index i = 0; i < m; ++i) {
                auto              udx       = static_cast<history_size_type>(i);
                const VectorReal &eigVals_i = v.at(udx);
                assert(eigVals_i.size() == n);
                y(i) = eigVals_i[j];
            }
            if(apply_log10)
                slopes(j) = get_slope(x, y.array().log10());
            else
                slopes(j) = get_slope(x, y);
        }

        return slopes;
    }

    template<typename Scalar>
    bool base<Scalar>::rNorms_have_saturated() {
        if(tol_stall_rnorm <= RealScalar{0}) return false;
        const auto min_history_size = std::min<size_t>(status.max_history_size, size_t{2});
        if(status.iter < static_cast<Eigen::Index>(min_history_size)) return false;
        if(status.rNorms_history.size() < static_cast<size_t>(min_history_size)) return false;

        VectorReal &vals           = status.rNorms;
        VectorReal  stds           = get_standard_deviations(status.rNorms_history, false);
        VectorReal  threshold      = tol_stall_rnorm * vals.cwiseMax(VectorReal::Constant(vals.size(), std::numeric_limits<RealScalar>::min()));
        VectorIdxT  stds_saturated = (stds.array() < threshold.array()).template cast<Eigen::Index>();
        return stds_saturated.all();
    }

    template<typename Scalar>
    bool base<Scalar>::eigVals_have_saturated() {
        if(tol_stall_evals <= RealScalar{0}) return false;
        const auto min_history_size = std::min<size_t>(status.max_history_size, size_t{2});
        if(status.iter < static_cast<Eigen::Index>(min_history_size)) return false;
        if(status.eigVals_history.size() < static_cast<size_t>(min_history_size)) return false;
        VectorReal vals           = status.eigVal.cwiseAbs().array() + eps;
        VectorReal stds           = get_standard_deviations(status.eigVals_history, false);
        VectorReal rels           = stds.cwiseQuotient(vals);
        VectorIdxT rels_saturated = (rels.array() < tol_stall_evals).template cast<Eigen::Index>();
        return rels_saturated.all();
    }

    template<typename Scalar>
    void base<Scalar>::updateStatus() {
        // Accumulate counters from the inner solver
        status.num_matvecs_total  += status.num_matvecs + status.num_matvecs_inner;
        status.num_precond_total  += status.num_precond + status.num_precond_inner;
        status.time_matvecs_total += status.time_matvecs.get_time() + status.time_matvecs_inner.get_time();
        status.time_precond_total += status.time_precond.get_time() + status.time_precond_inner.get_time();

        // Eigenvalues are sorted in ascending order.
        status.oldVal  = status.eigVal.topRows(nev);
        status.eigVal  = T_evals(status.optIdx).topRows(nev);
        status.absDiff = (status.eigVal - status.oldVal).cwiseAbs();

        VectorReal denom = (RealScalar{0.5} * (status.eigVal + status.oldVal).array().abs()).matrix();
        denom            = denom.cwiseMax(VectorReal::Constant(denom.size(), std::numeric_limits<RealScalar>::min()));
        status.relDiff   = status.absDiff.cwiseQuotient(denom);

        status.rNorms_history.push_back(status.rNorms.topRows(nev));
        status.eigVals_history.push_back(status.eigVal.topRows(nev));
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

        constexpr auto beta   = RealScalar{0.5f};
        VectorReal     rNorms = status.rNorms.topRows(nev);
        VectorReal     tols   = rNormTols();
        RealScalar     relGap = status.gap * get_op_norm_estimate(status.eigVal.size() > 0 ? std::optional<RealScalar>{status.eigVal(0)} : std::nullopt);
        if(rNorms.size() != tols.size()) throw std::logic_error("unequal rNorm and tolerance sizes");
        status.rNorm_below_rnormTol = (rNorms.array() < tols.array()).all();
        status.rNorm_below_gap      = rNorms.maxCoeff() < beta * relGap;

        if(status.rNorm_below_rnormTol) {
            std::string msg_rnorm_gap = fmt::format(" | gap {:.3e} (rel {:.3e})", status.gap, relGap);
            status.stopMessage.emplace_back(fmt::format("converged rNorm {} < tol {}{} | iters {} | mv {} | {:.3e} s",
                                                        format_vector(VectorReal(status.rNorms.topRows(nev))), format_vector(tols), msg_rnorm_gap,
                                                        status.iter + 1, status.num_matvecs_total, status.time_elapsed.get_time()));
            status.stopReason |= StopReason::converged;
        }

        if(max_iters >= 0 && status.iter + 1 >= max_iters) {
            status.stopMessage.emplace_back(
                fmt::format("iters ({}) >= maxiter ({}) | mv {} | {:.3e} s", status.iter + 1, max_iters, status.num_matvecs_total,
                            status.time_elapsed.get_time()));
            status.stopReason |= StopReason::max_iters;
        }

        if(max_matvecs >= 0 && status.num_matvecs_total >= max_matvecs) {
            status.stopMessage.emplace_back(
                fmt::format("num_matvecs_total ({}) >= max_matvecs ({}) | {:.3e} s", status.num_matvecs_total, max_matvecs,
                            status.time_elapsed.get_time()));
            status.stopReason |= StopReason::max_matvecs;
        }

        if(std::min(status.saturation_count_eigVal, status.saturation_count_rNorm) >= status.saturation_count_max) {
            status.stopMessage.emplace_back(fmt::format("saturation_count (eigVal {} rNorm {}) >= saturation_count_max ({}) | it {} | mv {} | {:.3e} s",
                                                        status.saturation_count_eigVal, status.saturation_count_rNorm, status.saturation_count_max,
                                                        status.iter + 1, status.num_matvecs_total, status.time_elapsed.get_time()));
            status.stopReason |= StopReason::ritz_value_stalled;
            status.stopReason |= StopReason::ritz_residual_stalled;
        }
        else if(status.saturation_count_eigVal >= status.saturation_count_max * 2) {
            status.stopMessage.emplace_back(fmt::format("saturation_count eigVal {} >= saturation_count_max ({}) * 2 | it {} | mv {} | {:.3e} s",
                                                        status.saturation_count_eigVal, status.saturation_count_max, status.iter + 1,
                                                        status.num_matvecs_total, status.time_elapsed.get_time()));
            status.stopReason |= StopReason::ritz_value_stalled;
        } else if(status.saturation_count_eigVal > 2 && status.saturation_count_rNorm >= status.saturation_count_max * 2) {
            // Probably eigVal is stuck in some kind of cycle.
            status.stopMessage.emplace_back(fmt::format("saturation_count rNorm {} >= saturation_count_max ({}) * 2 | it {} | mv {} | {:.3e} s",
                                                        status.saturation_count_rNorm, status.saturation_count_max, status.iter + 1,
                                                        status.num_matvecs_total, status.time_elapsed.get_time()));
            status.stopReason |= StopReason::ritz_residual_stalled;
        }
    }

    template<typename Scalar>
    void base<Scalar>::printStatus() {
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
                                   status.num_matvecs_inner, status.num_jdops_inner, status.num_precond_inner, status.inner_error_last,
                                   status.inner_tol_last, status.time_matvecs_inner.get_time(), status.time_jdops_inner.get_time(),
                                   status.time_precond_inner.get_time());
        }

        RealScalar orthError = RealScalar{0};
        if(Q.cols() > 0) {
            MatrixType Gram = use_b_inner_product && BQ.size() == Q.size() ? Q.adjoint() * BQ : Q.adjoint() * Q;
            Gram            = (Gram + Gram.adjoint()).eval() / RealScalar{2};
            orthError       = (Gram - MatrixType::Identity(Gram.rows(), Gram.cols())).norm();
        }

        std::string evMsg;
        if(is_generalized_problem() && V.cols() > 0 && AV.size() == V.size() && BV.size() == V.size()) {
            VectorReal VAV = (V.adjoint() * AV).diagonal().real();
            VectorReal VBV = (V.adjoint() * BV).diagonal().real();
            evMsg          = fmt::format(" {} / {}", format_vector(VAV, "{:.16f}"), format_vector(VBV, "{:.16f}"));
        }

        bool                      log_long_time    = last_log_time.get_lap() > 10.0;
        bool                      log_every_ten_it = (status.iter + 1) % 10 == 0;
        spdlog::level::level_enum loglevel         = spdlog::level::debug;
        if(log_every_ten_it || log_long_time) loglevel = spdlog::level::info;
        if(!eiglog->should_log(loglevel)) return;
        [[maybe_unused]] auto lap = last_log_time.restart_lap();

        eiglog->log(loglevel,
                    "it {:3} mv {:3} pc {:3} t {:.1e}s dim {} {}eigVal {}{} "
                    "oErr {:.3e} rNorms {} rNormTol {} tol {:.2e} (rel {:.2e}) "
                    "({:9.2e}/mv) sat {}:{}/{} col {:2} b {} ritz {} "
                    "op norm {:.2e} cond {:.2e} sens {:.2e}{}",
                    status.iter + 1, status.num_matvecs, status.num_precond, status.time_elapsed.get_time(), N, innerMsg,
                    format_vector(VectorReal(status.eigVal), "{:.16f}"), evMsg, orthError, format_vector(VectorReal(status.rNorms)),
                    format_vector(rNormTols(), "{:.3e}"), abstol, reltol, get_rNorms_log10_change_per_matvec(), status.saturation_count_eigVal,
                    status.saturation_count_rNorm, status.saturation_count_max, Q.cols(), b, enum2sv(ritz), status.op_norm_estimate, status.condition,
                    status.sensitivity, msg_rnorm_gap);
    }

    template<typename Scalar>
    result_view<Scalar> base<Scalar>::result() const {
        return result_view<Scalar>(*this);
    }

    template<typename Scalar>
    void base<Scalar>::clear_result() {
        status  = Status{};
        qBlocks = 0;
        T       = MatrixType{};
        Aproj   = MatrixType{};
        Bproj   = MatrixType{};
        W       = MatrixType{};
        Q       = MatrixType{};
        AQ      = MatrixType{};
        BQ      = MatrixType{};
        V       = MatrixType{};
        AV      = MatrixType{};
        BV      = MatrixType{};
        V_prev  = MatrixType{};
        K       = MatrixType{};
        K_prev  = MatrixType{};
        S       = MatrixType{};
        S1      = MatrixType{};
        S2      = MatrixType{};
        D       = MatrixType{};
        M       = MatrixType{};
        AM      = MatrixType{};
        BM      = MatrixType{};
        T_evals = VectorReal{};
        T1      = MatrixType{};
        T2      = MatrixType{};
        T_evecs = MatrixType{};
        hhqr    = Eigen::HouseholderQR<MatrixType>{};
    }

    template<typename Scalar>
    void base<Scalar>::step() {
        preamble();
        build();
        diagonalizeT();
        extractRitzVectors();
        updateStatus();
        printStatus();
        status.iter++;
    }

    template<typename Scalar>
    void base<Scalar>::run() {
        auto token_elapsed = status.time_elapsed.tic_token();
        init();
        printStatus();
        status.iter++;
        while(true) {
            step();
            if(status.stopReason != StopReason::none) break;
        }
    }

    template<typename Scalar>
    void base<Scalar>::set_maxPrevBlocks(Eigen::Index pb) {
        maxPrevBlocks = pb;
    }

    template<typename Scalar>
    void base<Scalar>::assert_allFinite(const Eigen::Ref<const MatrixType> &X, const std::source_location &location) {
        if(X.cols() == 0) return;
        if(!X.allFinite())
            throw std::runtime_error(fmt::format("{}:{}: {}: matrix has non-finite elements", location.file_name(), location.line(), location.function_name()));
    }

    template<typename Scalar>
    void base<Scalar>::assert_l2_orthonormal(const Eigen::Ref<const MatrixType> &X, const OrthMeta &m, const std::source_location &location) {
        if(X.cols() == 0) return;

        MatrixType Gram      = X.adjoint() * X;
        RealScalar orthError = (Gram - MatrixType::Identity(Gram.rows(), Gram.cols())).norm();
        RealScalar xnorm     = X.norm();
        RealScalar t_abs     = static_cast<RealScalar>(X.size()) * eps * (xnorm + xnorm);
        RealScalar maskTol   = std::isfinite(m.maskTol) ? m.maskTol : normTol * static_cast<RealScalar>(X.cols());
        RealScalar finalTol  = std::max({t_abs, normTol, maskTol}) * RealScalar{10};

        if(orthError > finalTol && eiglog)
            eiglog->warn("{}:{}: {}: matrix is not L2-orthonormal: error {:.5e} > tol {:.5e}", location.file_name(), location.line(),
                         location.function_name(), orthError, finalTol);
        if(orthError > RealScalar{1000} * finalTol)
            throw std::runtime_error(fmt::format("{}:{}: {}: matrix is not L2-orthonormal: error {:.5e} > tol {:.5e}", location.file_name(), location.line(),
                                                 location.function_name(), orthError, finalTol));
    }

    template<typename Scalar>
    void base<Scalar>::assert_l2_orthogonal(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &Y, const OrthMeta &m,
                                            const std::source_location &location) {
        if(X.cols() == 0 || Y.cols() == 0) return;
        if(m.mask.size() > 0 && m.mask.sum() == 0) return;

        MatrixType Gram      = X.adjoint() * Y;
        RealScalar orthError = Gram.norm();
        RealScalar xnorm     = X.norm();
        RealScalar ynorm     = Y.norm();
        RealScalar t_abs     = static_cast<RealScalar>(X.size()) * eps * (xnorm + ynorm);
        RealScalar maskTol   = std::isfinite(m.maskTol) ? m.maskTol : orthTol * static_cast<RealScalar>(X.cols());
        RealScalar finalTol  = std::max({t_abs, orthTol, maskTol}) * RealScalar{10};

        if(orthError > finalTol && eiglog)
            eiglog->warn("{}:{}: {}: matrices are not L2-orthogonal: error {:.5e} > tol {:.5e}", location.file_name(), location.line(),
                         location.function_name(), orthError, finalTol);
        if(orthError > RealScalar{1000} * finalTol)
            throw std::runtime_error(fmt::format("{}:{}: {}: matrices are not L2-orthogonal: error {:.5e} > tol {:.5e}", location.file_name(),
                                                 location.line(), location.function_name(), orthError, finalTol));
    }

    template<typename Scalar>
    void base<Scalar>::assert_bm_orthonormal(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &B_X, const OrthMeta &m,
                                             const std::source_location &location) {
        if(X.cols() == 0) return;
        if(X.cols() != B_X.cols() || X.rows() != B_X.rows())
            throw std::runtime_error(fmt::format("{}:{}: {}: X and B_X have incompatible dimensions", location.file_name(), location.line(),
                                                 location.function_name()));

        MatrixType G1        = X.adjoint() * B_X;
        MatrixType G2        = B_X.adjoint() * X;
        MatrixType Gram      = G1;
        MatrixType Gram_symm = (G1 + G2) * half;
        MatrixType Gram_skew = (G1 - G2) * half;
        MatrixType I         = MatrixType::Identity(Gram.rows(), Gram.cols());
        RealScalar orthError = (Gram - I).norm();
        RealScalar symmError = (Gram_symm - I).norm();
        RealScalar skewError = Gram_skew.norm();

        RealScalar xnorm   = X.norm();
        RealScalar bxnorm  = B_X.norm();
        RealScalar bnorm   = std::isfinite(status.op_norm_estimate) ? status.op_norm_estimate : RealScalar{1};
        RealScalar t_abs   = orthTol * static_cast<RealScalar>(X.cols()) * (xnorm + bxnorm);
        RealScalar bmTol   = orthTol * static_cast<RealScalar>(X.cols()) * bnorm;
        RealScalar maskTol = std::isfinite(m.maskTol) ? m.maskTol : orthTol;
        RealScalar finalTol = std::max({t_abs, orthTol, bmTol, maskTol}) * RealScalar{10};

        RealScalar error = std::max({orthError, symmError, skewError});
        if(error > finalTol && eiglog)
            eiglog->warn("{}:{}: {}: matrix is not B-orthonormal: error {:.5e} > tol {:.5e} | orth {:.5e} symm {:.5e} skew {:.5e}",
                         location.file_name(), location.line(), location.function_name(), error, finalTol, orthError, symmError, skewError);
        if(error > RealScalar{1000} * finalTol)
            throw std::runtime_error(fmt::format("{}:{}: {}: matrix is not B-orthonormal: error {:.5e} > tol {:.5e}", location.file_name(),
                                                 location.line(), location.function_name(), error, finalTol));
    }

    template<typename Scalar>
    void base<Scalar>::assert_bm_orthogonal(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &B_Y, const OrthMeta &m,
                                            const std::source_location &location) {
        if(X.cols() == 0 || B_Y.cols() == 0) return;

        MatrixType Gram      = X.adjoint() * B_Y;
        RealScalar orthError = Gram.norm();
        RealScalar xnorm     = X.norm();
        RealScalar bynorm    = B_Y.norm();
        RealScalar bnorm     = std::isfinite(status.op_norm_estimate) ? status.op_norm_estimate : RealScalar{1};
        RealScalar t_abs     = orthTol * static_cast<RealScalar>(X.cols()) * (xnorm + bynorm);
        RealScalar bmTol     = orthTol * static_cast<RealScalar>(X.cols()) * bnorm;
        RealScalar maskTol   = std::isfinite(m.maskTol) ? m.maskTol : orthTol;
        RealScalar finalTol  = std::max({t_abs, orthTol, bmTol, maskTol}) * RealScalar{10};

        if(orthError > finalTol && eiglog)
            eiglog->warn("{}:{}: {}: matrices are not B-orthogonal: error {:.5e} > tol {:.5e}", location.file_name(), location.line(),
                         location.function_name(), orthError, finalTol);
        if(orthError > RealScalar{1000} * finalTol)
            throw std::runtime_error(fmt::format("{}:{}: {}: matrices are not B-orthogonal: error {:.5e} > tol {:.5e}", location.file_name(),
                                                 location.line(), location.function_name(), orthError, finalTol));
    }

    template<typename Scalar>
    void base<Scalar>::compress_cols(MatrixType &X, const VectorIdxT &mask) {
        assert(mask.size() == X.cols() && "Mask size must match number of columns in X.");
        if(mask.sum() == X.cols()) return;

        std::vector<Eigen::Index> active_columns;
        active_columns.reserve(static_cast<typename std::vector<Eigen::Index>::size_type>(X.cols()));
        for(Eigen::Index j = 0; j < X.cols(); ++j) {
            if(mask(j) == 1) active_columns.push_back(j);
        }
        active_columns.shrink_to_fit();
        X = X(Eigen::placeholders::all, active_columns).eval();
    }

    template<typename Scalar>
    void base<Scalar>::block_l2_orthogonalize(const MatrixType &X, const MatrixType &AX, MatrixType &Y, MatrixType &AY, OrthMeta &m) {
        if(X.cols() == 0 || Y.cols() == 0) {
            AY.resizeLike(Y);
            return;
        }
        if(m.mask.size() > 0 && m.mask.sum() == 0) return;

        assert_allFinite(X);
        assert_allFinite(AX);
        assert_allFinite(Y);

        if(std::isnan(m.orthTol)) m.orthTol = normTol * static_cast<RealScalar>(Y.cols());

        m.Gram      = X.adjoint() * Y;
        m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt();
        m.orthError = m.Gram.size() > 0 ? m.Gram.norm() : 0;

        MatrixType Gxx = X.adjoint() * X;

        Eigen::Index maxReps = 2;
        Eigen::Index rep     = 0;
        for(rep = 0; rep < maxReps; ++rep) {
            MatrixType W  = Gxx.ldlt().solve(m.Gram);
            Y.noalias()  -= X * W;

            m.Gram      = X.adjoint() * Y;
            m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt();
            m.orthError = m.Gram.size() > 0 ? m.Gram.norm() : 0;

            bool orth_converged = m.orthError < m.orthTol;
            if(orth_converged || Y.cols() == 0) break;
        }
        if(eiglog && eiglog->should_log(spdlog::level::trace))
            eiglog->trace("rep {} orthError after l2 orthogonalization: {:.3e} | orthTol {:.3e}", rep, m.orthError, m.orthTol);
        assert_l2_orthogonal(X, Y, m);
        AY = MultA(Y);
    }

    template<typename Scalar>
    void base<Scalar>::block_l2_orthogonalize(const MatrixType &X, const MatrixType &AX, const MatrixType &BX, MatrixType &Y, MatrixType &AY, MatrixType &BY,
                                              OrthMeta &m) {
        if(X.cols() == 0 || Y.cols() == 0) {
            AY.resizeLike(Y);
            BY.resizeLike(Y);
            return;
        }
        if(m.mask.size() > 0 && m.mask.sum() == 0) return;

        assert_allFinite(X);
        assert_allFinite(AX);
        assert_allFinite(BX);
        assert_allFinite(Y);

        if(std::isnan(m.orthTol)) m.orthTol = orthTol * static_cast<RealScalar>(Y.cols());
        m.orthTol   = std::max(m.orthTol, orthTol * static_cast<RealScalar>(Y.cols()));
        m.Gram      = X.adjoint() * Y;
        m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt();
        m.orthError = m.Gram.size() > 0 ? m.Gram.norm() : 0;

        MatrixType Gxx = X.adjoint() * X;

        Eigen::Index maxReps = 2;
        Eigen::Index rep     = 0;
        for(rep = 0; rep < maxReps; ++rep) {
            MatrixType W  = Gxx.ldlt().solve(m.Gram);
            Y.noalias()  -= X * W;

            m.Gram      = X.adjoint() * Y;
            m.Rdiag     = m.Gram.diagonal().cwiseAbs().cwiseSqrt();
            m.orthError = m.Gram.size() > 0 ? m.Gram.norm() : 0;

            bool orth_converged = m.orthError < m.orthTol;
            if(orth_converged || Y.cols() == 0) break;
        }
        if(eiglog && eiglog->should_log(spdlog::level::trace))
            eiglog->trace("rep {} orthError after l2 orthogonalization: {:.3e} | orthTol {:.3e}", rep, m.orthError, m.orthTol);
        assert_l2_orthogonal(X, Y, m);

        AY = MultA(Y);
        BY = MultB(Y);
    }

    template<typename Scalar>
    void base<Scalar>::block_bm_orthogonalize(const MatrixType &X, const MatrixType &AX, const MatrixType &BX, MatrixType &Y, MatrixType &AY, MatrixType &BY,
                                              OrthMeta &m) {
        if(X.cols() == 0 || Y.cols() == 0) {
            AY.resizeLike(Y);
            BY.resizeLike(Y);
            return;
        }
        if(m.mask.size() > 0 && m.mask.sum() == 0) return;
        assert(is_generalized_problem() && use_b_inner_product && "block_bm_orthogonalize is for B inner product");

        assert_allFinite(X);
        assert_allFinite(AX);
        assert_allFinite(BX);
        assert_allFinite(Y);
        assert_bm_orthonormal(X, BX, OrthMeta{});

        if(std::isnan(m.orthTol)) m.orthTol = orthTol * static_cast<RealScalar>(X.cols());
        m.orthTol = std::max(m.orthTol, eps * std::sqrt(status.op_norm_estimate));
        if(!std::isfinite(m.orthTol))
            throw std::runtime_error(fmt::format("block_bm_orthogonalize: invalid orthTol {:.3e} | op_norm_estimate {:.3e}", m.orthTol,
                                                 status.op_norm_estimate));

        bool has_refreshed_by = false;
        if(m.refresh_by || Y.size() != BY.size()) {
            BY               = MultB(Y);
            has_refreshed_by = true;
            if(eiglog && eiglog->should_log(spdlog::level::trace)) eiglog->trace("block_bm_orthogonalize: refreshed BY");
        } else {
            assert_allFinite(BY);
        }

        m.analyze_bm_orthogonality(X, BX, Y, BY);

        MatrixType Gyy = Y.adjoint() * BY;
        Gyy            = (Gyy + Gyy.adjoint()).eval() * half;
        RealScalar Eyy = (Gyy - MatrixType::Identity(Gyy.cols(), Gyy.rows())).norm();

        MatrixType Gxx = X.adjoint() * BX;
        Gxx            = (Gxx + Gxx.adjoint()).eval() * half;
        RealScalar Exx = (Gxx - MatrixType::Identity(Gxx.cols(), Gxx.rows())).norm();

        if(m.skewError > std::sqrt(m.orthTol) && !has_refreshed_by) {
            MatrixType BY_new = MultB(Y);
            OrthMeta   m_new  = m;
            m_new.analyze_bm_orthogonality(X, BX, Y, BY_new);
            if(m_new.skewError < m.skewError) {
                BY.swap(BY_new);
                m                = m_new;
                has_refreshed_by = true;
            }
        }

        if(std::isfinite(m.orthTol) && std::max(m.symmError, m.skewError) < m.orthTol) {
            if(has_refreshed_by || m.refresh_by || Y.size() != AY.size()) AY = MultA(Y);
            if(eiglog && eiglog->should_log(spdlog::level::trace))
                eiglog->trace("block_bm_orthogonalize: no need: orthError {:.4e} symmError {:.4e} skewError {:.4e} Eyy {:.4e} orthTol {:.4e}",
                              m.orthError, m.symmError, m.skewError, Eyy, m.orthTol);
            return;
        }
        m.refresh_by = false;

        if(Exx > m.orthTol && eiglog && eiglog->should_log(spdlog::level::debug))
            eiglog->debug("block_bm_orthogonalize: X is not sufficiently B-orthonormal: error {:.4e}", Exx);

        Eigen::Index maxReps = 2;
        Eigen::Index rep     = 0;
        for(rep = 0; rep < maxReps; ++rep) {
            if(m.mask.size() != Y.cols()) m.mask = VectorIdxT::Ones(Y.cols());
            if(m.proj_sum_a.size() != Y.cols()) m.proj_sum_a = VectorReal::Zero(Y.cols());
            if(m.proj_sum_b.size() != Y.cols()) m.proj_sum_b = VectorReal::Zero(Y.cols());
            if(m.scale_log.size() != Y.cols()) m.scale_log = VectorReal::Zero(Y.cols());

            MatrixType W = Gxx.ldlt().solve(m.Gram_symm);

            Y.noalias()  -= X * W;
            BY.noalias() -= BX * W;

            m.analyze_bm_orthogonality(X, BX, Y, BY);
            if(rep >= 1) {
                bool orth_converged = std::max(m.symmError, m.skewError) < m.orthTol;
                if(orth_converged) break;
            }
        }
        if(eiglog && eiglog->should_log(spdlog::level::trace))
            eiglog->trace("rep {} orthError after bm orthogonalization: {:.3e} | symm {:.3e} | skew {:.3e} | orthTol {:.3e}", rep, m.orthError,
                          m.symmError, m.skewError, m.orthTol);

        AY = MultA(Y);
        assert_bm_orthogonal(X, BY, m);
        assert_bm_orthogonal(BX, Y, m);
    }

    template<typename Scalar>
    void base<Scalar>::block_l2_orthonormalize(MatrixType &Y, MatrixType &AY, OrthMeta &m) {
        if(Y.cols() == 0) {
            AY.resizeLike(Y);
            return;
        }
        if(m.mask.size() > 0 && m.mask.sum() == 0) return;

        assert(!use_b_inner_product);

        m.mask = VectorIdxT::Ones(Y.cols());
        if(std::isnan(m.maskTol)) m.maskTol = normTol * static_cast<RealScalar>(Y.cols());

        auto handle_masked_columns = [&]() {
            if(m.mask.sum() == Y.cols()) return;
            VectorReal norms = (Y.adjoint() * Y).diagonal().cwiseAbs();
            switch(m.maskPolicy) {
                case MaskPolicy::COMPRESS:
                    if(eiglog && eiglog->should_log(spdlog::level::debug))
                        eiglog->debug("block_l2_orthonormalize: compressing Y | kept {} of {} | maskTol {:.3e}", m.mask.sum(), Y.cols(), m.maskTol);
                    compress_cols(Y, m.mask);
                    m.mask = VectorIdxT::Ones(Y.cols());
                    break;
                case MaskPolicy::RANDOMIZE:
                    if(eiglog && eiglog->should_log(spdlog::level::debug))
                        eiglog->debug("block_l2_orthonormalize: randomizing masked columns | kept {} of {} | maskTol {:.3e}", m.mask.sum(), Y.cols(),
                                      m.maskTol);
                    for(Eigen::Index j = 0; j < Y.cols(); ++j) {
                        if(m.mask(j) == 0) Y.col(j) = Eigen::VectorXf::Random(Y.col(j).size()).template cast<Scalar>();
                    }
                    break;
            }
            (void) norms;
        };

        for(Eigen::Index j = 0; j < Y.cols(); ++j) {
            auto       yj   = Y.col(j);
            RealScalar norm = yj.norm();
            if(norm < m.maskTol) {
                if(eiglog && eiglog->should_log(spdlog::level::trace))
                    eiglog->trace("masking Y col {} | norm {:.3e} | maskTol {:.3e}", j, norm, m.maskTol);
                m.mask(j) = 0;
                yj.setZero();
            }
        }
        handle_masked_columns();
        if(Y.cols() == 0) {
            AY.resizeLike(Y);
            return;
        }

        hhqr.compute(Y);
        Y       = hhqr.householderQ().setLength(Y.cols()) * MatrixType::Identity(Y.rows(), Y.cols());
        m.Rdiag = hhqr.matrixQR().diagonal().cwiseAbs().topRows(Y.cols());

        for(Eigen::Index j = 0; j < Y.cols(); ++j) {
            auto       yj   = Y.col(j);
            RealScalar norm = yj.norm();
            if(norm < m.maskTol) {
                if(eiglog && eiglog->should_log(spdlog::level::trace))
                    eiglog->trace("masking Y col {} | norm {:.3e} | maskTol {:.3e}", j, norm, m.maskTol);
                m.mask(j) = 0;
                yj.setZero();
            }
        }
        handle_masked_columns();

        AY = MultA(Y);
        m.analyze_l2_orthonormality(Y);
        assert_l2_orthonormal(Y, m);
    }

    template<typename Scalar>
    void base<Scalar>::block_l2_orthonormalize(MatrixType &Y, MatrixType &AY, MatrixType &BY, OrthMeta &m) {
        assert(!use_b_inner_product);
        block_l2_orthonormalize(Y, AY, m);
        if(Y.cols() == 0) {
            BY.resizeLike(Y);
            return;
        }
        BY = MultB(Y);
    }

    template<typename Scalar>
    void base<Scalar>::block_bm_orthonormalize(MatrixType &Y, MatrixType &AY, MatrixType &BY, OrthMeta &m) {
        if(Y.cols() == 0) {
            AY.resizeLike(Y);
            BY.resizeLike(Y);
            return;
        }
        if(m.mask.size() > 0 && m.mask.sum() == 0) return;
        assert(is_generalized_problem() && use_b_inner_product && "block_bm_orthonormalize is for B inner product");

        auto handle_masked_columns = [&]() {
            if(m.mask.sum() == Y.cols()) return;
            switch(m.maskPolicy) {
                case MaskPolicy::COMPRESS:
                    if(eiglog && eiglog->should_log(spdlog::level::debug))
                        eiglog->debug("block_bm_orthonormalize: compressing Y | kept {} of {} | maskTol {:.3e}", m.mask.sum(), Y.cols(), m.maskTol);
                    compress_cols(Y, m.mask);
                    if(BY.rows() == Y.rows() && BY.cols() == m.mask.size()) compress_cols(BY, m.mask);
                    m.mask = VectorIdxT::Ones(Y.cols());
                    if(BY.rows() == Y.rows() && BY.cols() == Y.cols()) m.analyze_bm_orthonormality(Y, BY);
                    break;
                case MaskPolicy::RANDOMIZE:
                    if(eiglog && eiglog->should_log(spdlog::level::debug))
                        eiglog->debug("block_bm_orthonormalize: randomizing masked columns | kept {} of {} | maskTol {:.3e}", m.mask.sum(), Y.cols(),
                                      m.maskTol);
                    for(Eigen::Index j = 0; j < Y.cols(); ++j) {
                        if(m.mask(j) == 0) Y.col(j) = Eigen::VectorXf::Random(Y.col(j).size()).template cast<Scalar>();
                    }
                    BY = MultB(Y);
                    m.analyze_bm_orthonormality(Y, BY);
                    break;
            }
        };

        m.mask = VectorIdxT::Ones(Y.cols());
        m.proj_sum_b = VectorReal::Zero(Y.cols());
        m.scale_log  = VectorReal::Zero(Y.cols());
        if(std::isnan(m.maskTol)) m.maskTol = normTol * static_cast<RealScalar>(Y.cols());
        if(std::isnan(m.orthTol)) m.orthTol = normTol * static_cast<RealScalar>(Y.cols());

        if(m.refresh_by || Y.cols() != BY.cols() || Y.rows() != BY.rows()) BY = MultB(Y);
        m.refresh_by = false;
        m.analyze_bm_orthonormality(Y, BY);
        assert_allFinite(BY);

        m.Rdiag = VectorReal::Zero(Y.cols());
        for(Eigen::Index j = 0; j < Y.cols(); ++j) {
            auto       yj     = Y.col(j);
            auto       byj    = BY.col(j);
            RealScalar normSq = std::real(yj.dot(byj));
            RealScalar norm   = std::sqrt(std::max<RealScalar>(0, normSq));
            m.Rdiag(j)        = norm;
            if(norm < m.maskTol) {
                if(eiglog && eiglog->should_log(spdlog::level::trace))
                    eiglog->trace("masking Y col {} | bm norm {:.3e} | maskTol {:.3e}", j, norm, m.maskTol);
                m.mask(j) = 0;
                yj.setZero();
                byj.setZero();
            }
        }
        handle_masked_columns();
        if(Y.cols() == 0) {
            AY.resizeLike(Y);
            BY.resizeLike(Y);
            return;
        }

        Eigen::Index maxReps = 2;
        for(Eigen::Index rep = 0; rep < maxReps; ++rep) {
            VectorReal  normSqs = VectorReal::Zero(Y.cols());
            VectorIdxT  have    = VectorIdxT::Zero(Y.cols());

            for(Eigen::Index j = 0; j < Y.cols(); ++j) {
                if(m.mask(j) == 0) continue;

                auto yj  = Y.col(j);
                auto byj = BY.col(j);

                for(Eigen::Index i = 0; i < j; ++i) {
                    if(m.mask(i) == 0) continue;
                    auto yi  = Y.col(i);
                    auto byi = BY.col(i);

                    if(have(i) == 0) {
                        normSqs(i) = std::max<RealScalar>(0, std::real(yi.dot(byi)));
                        have(i)    = 1;
                    }

                    RealScalar normSq  = normSqs(i);
                    Scalar     proj1   = yi.dot(byj);
                    Scalar     proj2   = byi.dot(yj);
                    Scalar     proj_ij = normSq > std::numeric_limits<RealScalar>::min() ? (proj1 + proj2) / (RealScalar{2} * normSq) : Scalar{0};

                    yj.noalias()  -= yi * proj_ij;
                    byj.noalias() -= byi * proj_ij;
                }

                RealScalar normSq = std::real(yj.dot(byj));
                RealScalar norm   = std::sqrt(std::max<RealScalar>(0, normSq));
                if(norm <= m.maskTol) {
                    if(eiglog && eiglog->should_log(spdlog::level::trace))
                        eiglog->trace("masking Y col {} | bm norm {:.3e} | maskTol {:.3e}", j, norm, m.maskTol);
                    m.mask(j) = 0;
                    yj.setZero();
                    byj.setZero();
                    continue;
                }

                yj  /= norm;
                byj /= norm;
                normSqs(j) = std::max<RealScalar>(0, std::real(yj.dot(byj)));
            }

            m.analyze_bm_orthonormality(Y, BY);
            handle_masked_columns();
            if(Y.cols() == 0) {
                AY.resizeLike(Y);
                BY.resizeLike(Y);
                return;
            }
            if(eiglog && eiglog->should_log(spdlog::level::trace))
                eiglog->trace("block_bm_orthonormalize: dgks rep {} | orthError {:.4e} symmError {:.4e} skewError {:.4e}", rep, m.orthError,
                              m.symmError, m.skewError);
            if(m.orthError < m.orthTol) break;
        }

        AY = MultA(Y);
        assert_bm_orthonormal(Y, BY, m);
    }
}

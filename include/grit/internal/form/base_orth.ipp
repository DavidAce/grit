#pragma once

#include "grit/form/base.h"

namespace grit::form {
    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::RealScalar base<Scalar, form_>::bm_cancellation_multiplier(const Eigen::Ref<const MatrixType> &Y,
                                                                                             const Eigen::Ref<const MatrixType> &BY) const {
        if(Y.cols() == 0 || Y.rows() != BY.rows() || Y.cols() != BY.cols()) return RealScalar{1};

        RealScalar local_rayleigh_scale = std::numeric_limits<RealScalar>::infinity();
        for(Eigen::Index j = 0; j < Y.cols(); ++j) {
            const RealScalar ynorm_sq = std::real(Y.col(j).dot(Y.col(j)));
            const RealScalar denom    = std::max({std::abs(ynorm_sq), quotTolB, std::numeric_limits<RealScalar>::min()});
            const RealScalar numer    = std::abs(Y.col(j).dot(BY.col(j)));
            const RealScalar rq       = numer / denom;
            if(std::isfinite(rq) && rq > RealScalar{0}) local_rayleigh_scale = std::min(local_rayleigh_scale, rq);
        }
        if(!std::isfinite(local_rayleigh_scale) || local_rayleigh_scale <= RealScalar{0}) local_rayleigh_scale = RealScalar{1};

        const RealScalar ynorm          = Y.norm();
        const RealScalar bynorm         = BY.norm();
        const RealScalar observed_scale = ynorm > std::numeric_limits<RealScalar>::min() ? bynorm / ynorm : RealScalar{1};
        RealScalar       operator_scale = RealScalar{1};
        if constexpr(form_ == grit::Form::GENERALIZED) {
            if(B) operator_scale = B->get().get_op_norm();
        }

        operator_scale = std::max({operator_scale, status.op_norm_estimate, observed_scale, RealScalar{1}});
        if(!std::isfinite(operator_scale) || operator_scale <= RealScalar{0}) operator_scale = RealScalar{1};

        const RealScalar multiplier_max = RealScalar{1} / std::sqrt(eps);
        const RealScalar multiplier_raw = operator_scale / std::max(local_rayleigh_scale, std::numeric_limits<RealScalar>::min());

        return std::clamp(std::max(RealScalar{1}, multiplier_raw), RealScalar{1}, multiplier_max);
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::OrthMeta::analyze_l2_orthonormality(const Eigen::Ref<const MatrixType> &Y) {
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

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::OrthMeta::analyze_b_orthonormality(const Eigen::Ref<const MatrixType> &Y, const Eigen::Ref<const MatrixType> &BY) {
        analyze_bm_orthonormality(Y, BY);
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::OrthMeta::analyze_bm_orthonormality(const Eigen::Ref<const MatrixType> &Y, const Eigen::Ref<const MatrixType> &BY) {
        if(Y.cols() != BY.cols() || Y.rows() != BY.rows()) return;
        MatrixType I  = MatrixType::Identity(Y.cols(), Y.cols());
        MatrixType G1 = Y.adjoint() * BY;
        MatrixType G2 = BY.adjoint() * Y;
        Gram          = G1;
        Gram_symm     = (G1 + G2) * half;
        Gram_skew     = (G1 - G2) * half;
        orthError     = (Gram - I).norm();
        symmError     = (Gram_symm - I).norm();
        skewError     = Gram_skew.norm();
        skewError_fwd = skewError;
        Rdiag         = Gram_symm.diagonal().cwiseAbs().cwiseSqrt();
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::OrthMeta::analyze_l2_orthogonality(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &Y) {
        if(Y.cols() == 0) return;
        Gram      = X.adjoint() * Y;
        Gram_symm = Gram;
        Gram_skew = Gram;
        orthError = Gram.norm();
        symmError = orthError;
        skewError = orthError;
        Rdiag     = Gram_symm.diagonal().cwiseAbs().cwiseSqrt();
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::OrthMeta::analyze_bm_orthogonality(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &BX,
                                                                 const Eigen::Ref<const MatrixType> &Y, const Eigen::Ref<const MatrixType> &BY) {
        if(Y.cols() != BY.cols() || Y.rows() != BY.rows()) return;
        if(X.cols() != BX.cols() || X.rows() != BX.rows()) return;
        if(Y.rows() != X.rows()) return;

        MatrixType G1 = X.adjoint() * BY;
        MatrixType G2 = BX.adjoint() * Y;

        Gram      = G1;
        Gram_symm = (G1 + G2) * half;
        Gram_skew = (G1 - G2) * half;
        orthError = Gram_symm.norm();
        symmError = orthError;
        skewError = Gram_skew.norm();
        Rdiag     = Gram_symm.diagonal().cwiseAbs().cwiseSqrt();
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::assert_allFinite(const Eigen::Ref<const MatrixType> &X, const std::source_location &location) {
        if(X.cols() == 0) return;
        if(!X.allFinite())
            throw std::runtime_error(fmt::format("{}:{}: {}: matrix has non-finite elements", location.file_name(), location.line(), location.function_name()));
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::assert_l2_orthonormal(const Eigen::Ref<const MatrixType> &X, const OrthMeta &m, const std::source_location &location) {
        if(X.cols() == 0) return;

        MatrixType Gram      = X.adjoint() * X;
        RealScalar orthError = (Gram - MatrixType::Identity(Gram.rows(), Gram.cols())).norm();
        RealScalar xnorm     = X.norm();
        RealScalar t_abs     = static_cast<RealScalar>(X.size()) * eps * (xnorm + xnorm);
        RealScalar maskTol   = std::isfinite(m.maskTol) ? m.maskTol : normTol * static_cast<RealScalar>(X.cols());
        RealScalar finalTol  = std::max({t_abs, normTol, maskTol}) * RealScalar{10};

        if(orthError > finalTol && log)
            log->warn("{}:{}: {}: matrix is not L2-orthonormal: error {:.5e} > tol {:.5e}", location.file_name(), location.line(), location.function_name(),
                      orthError, finalTol);
        if(orthError > RealScalar{1000} * finalTol)
            throw std::runtime_error(fmt::format("{}:{}: {}: matrix is not L2-orthonormal: error {:.5e} > tol {:.5e}", location.file_name(), location.line(),
                                                 location.function_name(), orthError, finalTol));
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::assert_l2_orthogonal(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &Y, const OrthMeta &m,
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

        if(orthError > finalTol && log)
            log->warn("{}:{}: {}: matrices are not L2-orthogonal: error {:.5e} > tol {:.5e}", location.file_name(), location.line(), location.function_name(),
                      orthError, finalTol);
        if(orthError > RealScalar{1000} * finalTol)
            throw std::runtime_error(fmt::format("{}:{}: {}: matrices are not L2-orthogonal: error {:.5e} > tol {:.5e}", location.file_name(), location.line(),
                                                 location.function_name(), orthError, finalTol));
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::assert_bm_orthonormal(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &BX, const OrthMeta &m,
                                                    const std::source_location &location) {
        if(X.cols() == 0) return;
        if(X.cols() != BX.cols() || X.rows() != BX.rows())
            throw std::runtime_error(
                fmt::format("{}:{}: {}: X and BX have incompatible dimensions", location.file_name(), location.line(), location.function_name()));

        MatrixType G1        = X.adjoint() * BX;
        MatrixType G2        = BX.adjoint() * X;
        MatrixType Gram      = G1;
        MatrixType Gram_symm = (G1 + G2) * half;
        MatrixType Gram_skew = (G1 - G2) * half;
        MatrixType I         = MatrixType::Identity(Gram.rows(), Gram.cols());
        RealScalar orthError = (Gram - I).norm();
        RealScalar symmError = (Gram_symm - I).norm();
        RealScalar skewError = Gram_skew.norm();

        Eigen::SelfAdjointEigenSolver<MatrixType> esG(Gram_symm);

        VectorReal evG_abs = esG.eigenvalues().cwiseAbs();
        RealScalar evG_max = evG_abs.size() > 0 ? evG_abs.maxCoeff() : RealScalar{1};
        RealScalar evG_min = evG_abs.size() > 0 ? evG_abs.minCoeff() : RealScalar{1};
        evG_max            = std::max(evG_max, eps);
        evG_min            = std::max(evG_min, eps);

        RealScalar one     = RealScalar{1};
        RealScalar xnorm   = X.norm();
        RealScalar bxnorm  = BX.norm();
        RealScalar xrows   = static_cast<RealScalar>(X.rows());
        RealScalar gamma_n = xrows * eps / std::max(one - xrows * eps, eps);
        // The raw dot-product floor assumes B X lives at the same scale as X.
        // Inflate it when the B-product is cancellation dominated.
        const RealScalar cancellation_multiplier = bm_cancellation_multiplier(X, BX);
        const RealScalar dotTol                  = gamma_n * cancellation_multiplier * xnorm * bxnorm;
        const RealScalar kappaG                  = evG_max / evG_min;
        const RealScalar kappaGTol               = RealScalar{20} * eps * kappaG;
        const RealScalar maskTol                 = std::isfinite(m.maskTol) ? m.maskTol : orthTol;
        const RealScalar finalTol                = std::max({orthTol, dotTol, kappaGTol, maskTol}) * RealScalar{10};

        if(skewError > RealScalar{1e-2f} && log) {
            log->warn("{}:{}: {}: skew-Hermitian B Gram diagnostic {:.5e} | orth {:.5e} symm {:.5e}", location.file_name(), location.line(),
                      location.function_name(), skewError, orthError, symmError);
        }
        if(symmError > finalTol && log) {
            log->warn("{}:{}: {}: matrix is not B-orthonormal: error {:.5e} > tol {:.5e} | symm {:.5e} skew {:.5e} | dotTol {:.5e} kappaG {:.5e}",
                      location.file_name(), location.line(), location.function_name(), orthError, finalTol, symmError, skewError, dotTol, kappaG);
        }
        if(symmError > finalTol && orthError > RealScalar{1000} * finalTol)
            throw std::runtime_error(fmt::format("{}:{}: {}: matrix is not B-orthonormal: error {:.5e} > tol {:.5e}", location.file_name(), location.line(),
                                                 location.function_name(), orthError, finalTol));
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::assert_bm_orthogonal(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &BX,
                                                   const Eigen::Ref<const MatrixType> &Y, const Eigen::Ref<const MatrixType> &BY, const OrthMeta &m,
                                                   const std::source_location &location) {
        if(X.cols() == 0 || BY.cols() == 0) return;
        if(X.cols() != BX.cols() || X.rows() != BX.rows())
            throw std::runtime_error(
                fmt::format("{}:{}: {}: X and BX have incompatible dimensions", location.file_name(), location.line(), location.function_name()));
        if(Y.cols() != BY.cols() || Y.rows() != BY.rows())
            throw std::runtime_error(
                fmt::format("{}:{}: {}: Y and BY have incompatible dimensions", location.file_name(), location.line(), location.function_name()));

        MatrixType Gram      = X.adjoint() * BY;
        RealScalar orthError = Gram.norm();
        RealScalar one       = RealScalar{1};
        RealScalar xrows     = static_cast<RealScalar>(X.rows());
        RealScalar xnorm     = X.norm();
        RealScalar bynorm    = BY.norm();
        RealScalar gamma_n   = xrows * eps / std::max(one - xrows * eps, eps);
        ;
        // Same idea as in the orthonormality check: inflate the dot-product floor
        // when the local B-product appears to be cancellation dominated.
        const RealScalar cancellation_multiplier = bm_cancellation_multiplier(Y, BY);
        const RealScalar dotTol                  = gamma_n * cancellation_multiplier * xnorm * bynorm;
        const RealScalar maskTol                 = std::isfinite(m.maskTol) ? m.maskTol : orthTol;
        const RealScalar finalTol                = std::max({orthTol, dotTol, maskTol}) * RealScalar{10};
        RealScalar       metric_cond             = RealScalar{1};

        if(orthError > finalTol && log) {
            MatrixType Gxx = X.adjoint() * BX;
            Gxx            = (Gxx + Gxx.adjoint()).eval() * half;
            Eigen::SelfAdjointEigenSolver<MatrixType> esG(Gxx);
            if(esG.info() == Eigen::Success && esG.eigenvalues().size() > 0) {
                VectorReal ev_abs = esG.eigenvalues().cwiseAbs();
                RealScalar ev_max = std::max(ev_abs.maxCoeff(), eps);
                RealScalar ev_min = std::max(ev_abs.minCoeff(), eps);
                metric_cond       = ev_max / ev_min;
            }
            log->warn("{}:{}: {}: matrices are not B-orthogonal: error {:.5e} > tol {:.5e} | dotTol {:.5e} kappaG {:.5e}", location.file_name(),
                      location.line(), location.function_name(), orthError, finalTol, dotTol, metric_cond);
        }
        MatrixType reverse_gram   = BX.adjoint() * Y;
        RealScalar reverse_err    = reverse_gram.norm();
        RealScalar reverse_dotTol = gamma_n * BX.norm() * Y.norm();
        RealScalar reverse_tol    = RealScalar{10} * std::max(m.orthTol, reverse_dotTol * cancellation_multiplier);
        if(reverse_err > reverse_tol && log) {
            log->warn("{}:{}: {}: reverse B-orthogonality diagnostic {:.5e} > tol {:.5e} | dotTol {:.5e} cancel_mult {:.5e}", location.file_name(),
                      location.line(), location.function_name(), reverse_err, reverse_tol, reverse_dotTol, cancellation_multiplier);
        }
        if constexpr(grit::settings::debug_ortho) {
            if(orthError > RealScalar{1000} * finalTol)
                throw std::runtime_error(fmt::format("{}:{}: {}: matrices are not B-orthogonal: error {:.5e} > tol {:.5e}", location.file_name(),
                                                     location.line(), location.function_name(), orthError, finalTol));
        }
        if(!std::isfinite(orthError))
            throw std::runtime_error(fmt::format("{}:{}: {}: matrices are not B-orthogonal: error {:.5e} > tol {:.5e}", location.file_name(), location.line(),
                                                 location.function_name(), orthError, finalTol));
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::compress_cols(MatrixType &X, const VectorIdxT &mask) {
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

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::block_l2_orthogonalize(const MatrixType &X, const MatrixType &AX, MatrixType &Y, MatrixType &AY, OrthMeta &m,
                                                     RefreshMult refresh_mult) {
        auto           token_orthogonalize = status.time_orthogonalize.tic_token();
        constexpr auto inv_sqrt_2          = std::numbers::sqrt2_v<RealScalar> / 2;
        if(X.cols() == 0 || Y.cols() == 0) {
            AY.resizeLike(Y);
            return;
        }
        if(m.mask.size() > 0 && m.mask.sum() == 0) return;

        assert_allFinite(X);
        assert_allFinite(AX);
        assert_allFinite(Y);

        if(std::isnan(m.orthTol)) m.orthTol = normTol * static_cast<RealScalar>(Y.cols());

        {
            auto token_orth_project = status.time_orth_project.tic_token();
            m.Gram                  = X.adjoint() * Y;
            m.Rdiag                 = m.Gram.diagonal().cwiseAbs().cwiseSqrt();
            m.orthError             = m.Gram.size() > 0 ? m.Gram.norm() : 0;
        }

        MatrixType Gxx;
        {
            auto token_orth_project = status.time_orth_project.tic_token();
            Gxx                     = X.adjoint() * X;
        }

        VectorReal   ynorms0 = Y.colwise().norm().transpose();
        Eigen::Index rep     = 0;
        for(rep = 0; rep < 2; ++rep) {
            MatrixType W;
            {
                auto token_orth_factor = status.time_orth_factor.tic_token();
                W                      = Gxx.ldlt().solve(m.Gram);
            }
            {
                auto token_orth_update  = status.time_orth_update.tic_token();
                Y.noalias()            -= X * W;
            }

            VectorReal ynorms1 = Y.colwise().norm().transpose();
            {
                auto token_orth_project = status.time_orth_project.tic_token();
                m.Gram                  = X.adjoint() * Y;
                m.Rdiag                 = m.Gram.diagonal().cwiseAbs().cwiseSqrt();
                m.orthError             = m.Gram.size() > 0 ? m.Gram.norm() : 0;
            }

            bool orth_converged = m.orthError < m.orthTol;
            bool need_reorth    = (ynorms1.array() < inv_sqrt_2 * ynorms0.array()).any();
            if(rep == 0 && !need_reorth) break;
            if(orth_converged || Y.cols() == 0 || rep == 1) break;
            ynorms0 = ynorms1;
        }
        if constexpr(grit::settings::debug_ortho) {
            if(log && log->should_log(spdlog::level::trace))
                log->trace("rep {} orthError after l2 orthogonalization: {:.3e} | orthTol {:.3e}", rep, m.orthError, m.orthTol);
        }
        assert_l2_orthogonal(X, Y, m);
        if(refresh_mult == RefreshMult::YES) {
            auto token_orth_refresh = status.time_orth_refresh.tic_token();
            AY                      = MultA(Y);
        }
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::block_l2_orthogonalize(const MatrixType &X, const MatrixType &AX, const MatrixType &BX, MatrixType &Y, MatrixType &AY,
                                                     MatrixType &BY, OrthMeta &m, RefreshMult refresh_mult) {
        auto           token_orthogonalize = status.time_orthogonalize.tic_token();
        constexpr auto inv_sqrt_2          = std::numbers::sqrt2_v<RealScalar> / 2;
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
        m.orthTol = std::max(m.orthTol, orthTol * static_cast<RealScalar>(Y.cols()));
        {
            auto token_orth_project = status.time_orth_project.tic_token();
            m.Gram                  = X.adjoint() * Y;
            m.Rdiag                 = m.Gram.diagonal().cwiseAbs().cwiseSqrt();
            m.orthError             = m.Gram.size() > 0 ? m.Gram.norm() : 0;
        }

        MatrixType Gxx;
        {
            auto token_orth_project = status.time_orth_project.tic_token();
            Gxx                     = X.adjoint() * X;
        }

        VectorReal   ynorms0 = Y.colwise().norm().transpose();
        Eigen::Index rep     = 0;
        for(rep = 0; rep < 2; ++rep) {
            MatrixType W;
            {
                auto token_orth_factor = status.time_orth_factor.tic_token();
                W                      = Gxx.ldlt().solve(m.Gram);
            }
            {
                auto token_orth_update  = status.time_orth_update.tic_token();
                Y.noalias()            -= X * W;
            }

            VectorReal ynorms1 = Y.colwise().norm().transpose();
            {
                auto token_orth_project = status.time_orth_project.tic_token();
                m.Gram                  = X.adjoint() * Y;
                m.Rdiag                 = m.Gram.diagonal().cwiseAbs().cwiseSqrt();
                m.orthError             = m.Gram.size() > 0 ? m.Gram.norm() : 0;
            }

            bool orth_converged = m.orthError < m.orthTol;
            bool need_reorth    = (ynorms1.array() < inv_sqrt_2 * ynorms0.array()).any();
            if(rep == 0 && !need_reorth) break;
            if(orth_converged || Y.cols() == 0 || rep == 1) break;
            ynorms0 = ynorms1;
        }
        if(log && log->should_log(spdlog::level::trace))
            log->trace("rep {} orthError after l2 orthogonalization: {:.3e} | orthTol {:.3e}", rep, m.orthError, m.orthTol);
        assert_l2_orthogonal(X, Y, m);

        if(refresh_mult == RefreshMult::YES) {
            auto token_orth_refresh = status.time_orth_refresh.tic_token();
            AY                      = MultA(Y);
            if constexpr(form_ == grit::Form::GENERALIZED)
                BY = MultB(Y);
            else
                BY = Y;
        }
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::block_bm_orthogonalize(const MatrixType &X, const MatrixType &AX, const MatrixType &BX, MatrixType &Y, MatrixType &AY,
                                                     MatrixType &BY, OrthMeta &m, RefreshMult refresh_mult) requires(form_ == grit::Form::GENERALIZED)
    {
        auto                 token_orthogonalize = status.time_orthogonalize.tic_token();
        constexpr RealScalar inv_sqrt_2          = static_cast<RealScalar>(0.70710678118654752440);
        if(X.cols() == 0 || Y.cols() == 0) {
            AY.resizeLike(Y);
            BY.resizeLike(Y);
            return;
        }
        if(m.mask.size() > 0 && m.mask.sum() == 0) return;
        assert(cfg().use_b_inner_product && "block_bm_orthogonalize is for B inner product");

        assert_allFinite(X);
        assert_allFinite(AX);
        assert_allFinite(BX);
        assert_allFinite(Y);
        assert_bm_orthonormal(X, BX, OrthMeta{});

        if(std::isnan(m.orthTol)) m.orthTol = orthTol * static_cast<RealScalar>(X.cols());
        m.orthTol = std::max(m.orthTol, eps * std::sqrt(status.op_norm_estimate));
        if(!std::isfinite(m.orthTol))
            throw std::runtime_error(
                fmt::format("block_bm_orthogonalize: invalid orthTol {:.3e} | op_norm_estimate {:.3e}", m.orthTol, status.op_norm_estimate));

        bool has_refreshed_by = false;
        if(m.refresh_by || Y.size() != BY.size()) {
            auto token_orth_refresh = status.time_orth_refresh.tic_token();
            BY                      = MultB(Y);
            has_refreshed_by        = true;
            if(log && log->should_log(spdlog::level::trace)) log->trace("block_bm_orthogonalize: refreshed BY");
        } else {
            assert_allFinite(BY);
        }

        {
            auto token_orth_project = status.time_orth_project.tic_token();
            m.analyze_bm_orthogonality(X, BX, Y, BY);
        }

        MatrixType Gyy;
        RealScalar Eyy;
        {
            auto token_orth_project = status.time_orth_project.tic_token();
            Gyy                     = Y.adjoint() * BY;
            Gyy                     = (Gyy + Gyy.adjoint()).eval() * half;
            Eyy                     = (Gyy - MatrixType::Identity(Gyy.cols(), Gyy.rows())).norm();
        }

        MatrixType Gxx;
        RealScalar Exx;
        {
            auto token_orth_project = status.time_orth_project.tic_token();
            Gxx                     = X.adjoint() * BX;
            Gxx                     = (Gxx + Gxx.adjoint()).eval() * half;
            Exx                     = (Gxx - MatrixType::Identity(Gxx.cols(), Gxx.rows())).norm();
        }

        if(m.skewError > std::sqrt(m.orthTol) && !has_refreshed_by) {
            MatrixType BY_new;
            {
                auto token_orth_refresh = status.time_orth_refresh.tic_token();
                BY_new                  = MultB(Y);
            }
            OrthMeta m_new = m;
            {
                auto token_orth_project = status.time_orth_project.tic_token();
                m_new.analyze_bm_orthogonality(X, BX, Y, BY_new);
            }
            if(m_new.skewError < m.skewError) {
                BY.swap(BY_new);
                m                = m_new;
                has_refreshed_by = true;
            }
        }

        if(std::isfinite(m.orthTol) && std::max(m.symmError, m.skewError) < m.orthTol) {
            if(has_refreshed_by || m.refresh_by || Y.size() != AY.size()) {
                auto token_orth_refresh = status.time_orth_refresh.tic_token();
                AY                      = MultA(Y);
            }
            if(log && log->should_log(spdlog::level::trace))
                log->trace("block_bm_orthogonalize: no need: orthError {:.4e} symmError {:.4e} skewError {:.4e} Eyy {:.4e} orthTol {:.4e}", m.orthError,
                           m.symmError, m.skewError, Eyy, m.orthTol);
            return;
        }
        m.refresh_by = false;

        if(Exx > m.orthTol && log && log->should_log(spdlog::level::debug))
            log->debug("block_bm_orthogonalize: X is not sufficiently B-orthonormal: error {:.4e}", Exx);

        VectorReal ynorms0 = VectorReal::Zero(Y.cols());
        for(Eigen::Index j = 0; j < Y.cols(); ++j) {
            RealScalar ynorm_sq = std::real(Y.col(j).dot(BY.col(j)));
            ynorms0(j)          = std::sqrt(std::max<RealScalar>(0, ynorm_sq));
        }

        Eigen::Index rep = 0;
        for(rep = 0; rep < 2; ++rep) {
            if(m.mask.size() != Y.cols()) m.mask = VectorIdxT::Ones(Y.cols());
            if(m.proj_sum_a.size() != Y.cols()) m.proj_sum_a = VectorReal::Zero(Y.cols());
            if(m.proj_sum_b.size() != Y.cols()) m.proj_sum_b = VectorReal::Zero(Y.cols());
            if(m.scale_log.size() != Y.cols()) m.scale_log = VectorReal::Zero(Y.cols());

            MatrixType W;
            {
                auto token_orth_factor = status.time_orth_factor.tic_token();
                W                      = Gxx.ldlt().solve(m.Gram);
            }

            {
                auto token_orth_update  = status.time_orth_update.tic_token();
                Y.noalias()            -= X * W;
                BY.noalias()           -= BX * W;
            }

            {
                auto token_orth_project = status.time_orth_project.tic_token();
                m.analyze_bm_orthogonality(X, BX, Y, BY);
            }
            if(m.Gram.norm() >= m.orthTol || m.skewError > std::sqrt(m.orthTol)) {
                auto token_orth_refresh = status.time_orth_refresh.tic_token();
                BY                      = MultB(Y);
                m.analyze_bm_orthogonality(X, BX, Y, BY);
            }

            VectorReal ynorms1 = VectorReal::Zero(Y.cols());
            for(Eigen::Index j = 0; j < Y.cols(); ++j) {
                RealScalar ynorm_sq = std::real(Y.col(j).dot(BY.col(j)));
                ynorms1(j)          = std::sqrt(std::max<RealScalar>(0, ynorm_sq));
            }

            bool orth_converged = std::max(m.symmError, m.skewError) < m.orthTol;
            bool need_reorth    = (ynorms1.array() < inv_sqrt_2 * ynorms0.array()).any();
            if(rep == 0 && !need_reorth) break;
            if(orth_converged || rep == 1) break;
            ynorms0 = ynorms1;
        }
        if(log && log->should_log(spdlog::level::trace))
            log->trace("rep {} orthError after bm orthogonalization: {:.3e} | symm {:.3e} | skew {:.3e} | orthTol {:.3e}", rep, m.orthError, m.symmError,
                       m.skewError, m.orthTol);

        if(refresh_mult == RefreshMult::YES) {
            auto token_orth_refresh = status.time_orth_refresh.tic_token();
            AY                      = MultA(Y);
        }
        assert_bm_orthogonal(X, BX, Y, BY, m);
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::block_l2_orthonormalize(MatrixType &Y, MatrixType &AY, OrthMeta &m) {
        auto token_orthonormalize = status.time_orthonormalize.tic_token();
        if(Y.cols() == 0) {
            AY.resizeLike(Y);
            return;
        }
        if(m.mask.size() > 0 && m.mask.sum() == 0) return;

        assert(!cfg().use_b_inner_product);

        m.mask = VectorIdxT::Ones(Y.cols());
        if(std::isnan(m.maskTol)) m.maskTol = normTol * static_cast<RealScalar>(Y.cols());

        auto handle_masked_columns = [&]() {
            if(m.mask.sum() == Y.cols()) return;
            VectorReal norms = (Y.adjoint() * Y).diagonal().cwiseAbs();
            switch(m.maskPolicy) {
                case MaskPolicy::COMPRESS:
                    if(log && log->should_log(spdlog::level::debug))
                        log->debug("block_l2_orthonormalize: compressing Y | kept {} of {} | maskTol {:.3e}", m.mask.sum(), Y.cols(), m.maskTol);
                    compress_cols(Y, m.mask);
                    m.mask = VectorIdxT::Ones(Y.cols());
                    break;
                case MaskPolicy::RANDOMIZE:
                    if(log && log->should_log(spdlog::level::debug))
                        log->debug("block_l2_orthonormalize: randomizing masked columns | kept {} of {} | maskTol {:.3e}", m.mask.sum(), Y.cols(), m.maskTol);
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
                if(log && log->should_log(spdlog::level::trace)) log->trace("masking Y col {} | norm {:.3e} | maskTol {:.3e}", j, norm, m.maskTol);
                m.mask(j) = 0;
                yj.setZero();
            }
        }
        {
            auto token_orth_mask = status.time_orth_mask.tic_token();
            handle_masked_columns();
        }
        if(Y.cols() == 0) {
            AY.resizeLike(Y);
            return;
        }

        {
            auto token_orth_factor = status.time_orth_factor.tic_token();
            hhqr.compute(Y);
            Y       = hhqr.householderQ().setLength(Y.cols()) * MatrixType::Identity(Y.rows(), Y.cols());
            m.Rdiag = hhqr.matrixQR().diagonal().cwiseAbs().topRows(Y.cols());
        }

        for(Eigen::Index j = 0; j < Y.cols(); ++j) {
            auto       yj   = Y.col(j);
            RealScalar norm = yj.norm();
            if(norm < m.maskTol) {
                if(log && log->should_log(spdlog::level::trace)) log->trace("masking Y col {} | norm {:.3e} | maskTol {:.3e}", j, norm, m.maskTol);
                m.mask(j) = 0;
                yj.setZero();
            }
        }
        {
            auto token_orth_mask = status.time_orth_mask.tic_token();
            handle_masked_columns();
        }

        {
            auto token_orth_refresh = status.time_orth_refresh.tic_token();
            AY                      = MultA(Y);
        }
        m.analyze_l2_orthonormality(Y);
        assert_l2_orthonormal(Y, m);
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::block_l2_orthonormalize(MatrixType &Y, MatrixType &AY, MatrixType &BY, OrthMeta &m) {
        assert(!cfg().use_b_inner_product);
        block_l2_orthonormalize(Y, AY, m);
        if(Y.cols() == 0) {
            BY.resizeLike(Y);
            return;
        }
        {
            auto token_orth_refresh = status.time_orth_refresh.tic_token();
            if constexpr(form_ == grit::Form::GENERALIZED)
                BY = MultB(Y);
            else
                BY = Y;
        }
    }

    template<typename LScalar>
    struct BmEigOrthoStepMeta {
        using RealLScalar = decltype(std::real(std::declval<LScalar>()));
        using MatrixLType = Eigen::Matrix<LScalar, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorLReal = Eigen::Matrix<RealLScalar, Eigen::Dynamic, 1>;
        MatrixLType Y, BY;
        MatrixLType G;
        RealLScalar symmError;
        template<typename Scalar>
        BmEigOrthoStepMeta(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &Y_Scalar, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &BY_Scalar)
            : Y(Y_Scalar.template cast<LScalar>()), BY(BY_Scalar.template cast<LScalar>()) {}
    };

    template<typename Scalar, typename RealScalar, typename LScalar>
    void do_bm_eig_orthonormalization_step(
        BmEigOrthoStepMeta<LScalar> &m,
        std::function<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &)>
                                                     applyB,
        [[maybe_unused]] const Logger::LoggerHandle &log) {
        using RealLScalar = typename BmEigOrthoStepMeta<LScalar>::RealLScalar;
        using MatrixLType = typename BmEigOrthoStepMeta<LScalar>::MatrixLType;
        using VectorLReal = typename BmEigOrthoStepMeta<LScalar>::VectorLReal;

        auto &G         = m.G;
        auto &Y         = m.Y;
        auto &BY        = m.BY;
        auto &symmError = m.symmError;

        static constexpr auto half = RealLScalar{1} / RealLScalar{2};

        auto assert_finite = [&]() {
            bool ynan  = !Y.allFinite();
            bool bynan = !BY.allFinite();
            if(ynan) throw std::runtime_error("do_bm_eig_orthonormalization_step: Y has nan or inf");
            if(bynan) throw std::runtime_error("do_bm_eig_orthonormalization_step: BY has nan or inf");
        };
        MatrixLType G1          = Y.adjoint() * BY;
        G                       = (G1 + G1.adjoint()) * half; // The Gram matrix must be Hermitian and positive semi-definite.
        symmError               = (G - MatrixLType::Identity(G.rows(), G.cols())).norm();
        VectorLReal Gdiag       = G.real().diagonal();
        VectorLReal scaleErrors = Gdiag - VectorLReal::Ones(Gdiag.size());
        assert_finite();
        if(log && log->should_log(spdlog::level::trace))
            log->trace("do_bm_eig_orthonormalization_step: max diag(G)-I scale error {:.5e}",
                       scaleErrors.size() > 0 ? scaleErrors.cwiseAbs().maxCoeff() : RealLScalar{0});

        if(Y.cols() == 0) {
            if(log && log->should_log(spdlog::level::trace)) log->trace("do_bm_eig_orthonormalization_step: no columns left");
            G         = MatrixLType();
            symmError = RealLScalar{0};
            return;
        }

        auto esG = Eigen::SelfAdjointEigenSolver<MatrixLType>(G);
        if(esG.info() != Eigen::Success) throw std::runtime_error("do_bm_eig_orthonormalization_step: eig failed");
        VectorLReal lG = esG.eigenvalues();
        if(log && log->should_log(spdlog::level::trace))
            log->trace("do_bm_eig_orthonormalization_step: Gram eigenvalue range [{:.5e}, {:.5e}]", lG.minCoeff(), lG.maxCoeff());

        RealLScalar eps100 = std::numeric_limits<RealLScalar>::epsilon() * RealLScalar{100};
        RealLScalar tol    = eps100 * std::max<RealLScalar>(RealLScalar{1}, lG.cwiseAbs().maxCoeff());

        std::vector<Eigen::Index> keep;
        for(Eigen::Index j = 0; j < lG.size(); ++j) {
            if(lG(j) > tol) {
                keep.push_back(j);
            } else if(log && log->should_log(spdlog::level::trace)) {
                log->trace("do_bm_eig_orthonormalization_step: dropping Gram eigenvalue {} of {}: {:.5e}", j, G.rows(), lG(j));
            }
        }

        if(keep.empty()) {
            Y.resize(Y.rows(), 0);
            BY.resize(BY.rows(), 0);
            G         = MatrixLType();
            symmError = RealLScalar{0};
            return;
        }

        VectorLReal D = lG(keep);
        MatrixLType U = esG.eigenvectors()(Eigen::placeholders::all, keep);
        MatrixLType W = U * D.cwiseInverse().cwiseSqrt().asDiagonal();

        Y  = (Y * W).eval();
        BY = (BY * W).eval();
        if(log && log->should_log(spdlog::level::debug)) {
            MatrixLType BYW       = applyB(Y.template cast<Scalar>()).template cast<LScalar>();
            MatrixLType Delta     = BYW - BY;
            MatrixLType E_predict = Y.adjoint() * Delta;
            log->debug("do_bm_eig_orthonormalization_step: refreshed BY mismatch {:.4e} | predicted Gram error {:.4e}", Delta.norm(), E_predict.norm());
        }

        G1 = Y.adjoint() * BY;
        G  = (G1 + G1.adjoint()) * half;

        MatrixLType E_BY_W = (G - MatrixLType::Identity(G.rows(), G.cols()));
        symmError          = E_BY_W.norm();
        assert_finite();
        if(log && log->should_log(spdlog::level::debug)) log->debug("do_bm_eig_orthonormalization_step: symmError {:.5e}", symmError);
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::block_bm_orthonormalize_eig(MatrixType &Y, MatrixType &AY, MatrixType &BY, OrthMeta &m) requires(form_ == grit::Form::GENERALIZED)
    {
        auto token_orthonormalize = status.time_orthonormalize.tic_token();
        if(Y.cols() == 0) {
            AY.resizeLike(Y);
            BY.resizeLike(Y);
            return;
        }
        if(m.mask.size() > 0 && m.mask.sum() == 0) return;

        assert(cfg().use_b_inner_product && "block_bm_orthonormalize_eig is for B inner product");
        assert(m.maskPolicy == MaskPolicy::COMPRESS);

        {
            auto token_orth_project = status.time_orth_project.tic_token();
            m.analyze_bm_orthonormality(Y, BY);
        }
        if(m.refresh_by || Y.cols() != BY.cols() || Y.rows() != BY.rows() || m.skewError > m.skewTol) {
            auto token_orth_refresh = status.time_orth_refresh.tic_token();
            BY                      = MultB(Y);
            m.analyze_bm_orthonormality(Y, BY);
            if(log && log->should_log(spdlog::level::debug)) log->debug("block_bm_orthonormalize_eig: refreshed BY");
        } else {
            assert_allFinite(BY);
        }

        m.refresh_by = false;

        if(log && log->should_log(spdlog::level::trace))
            log->trace("block_bm_orthonormalize_eig: initial orthError {:.4e} symmError {:.4e} skewError {:.4e}", m.orthError, m.symmError, m.skewError);

        if(std::isnan(m.orthTol)) m.orthTol = normTol * static_cast<RealScalar>(Y.cols());
        if(m.symmError < m.orthTol) {
            auto token_orth_refresh = status.time_orth_refresh.tic_token();
            AY                      = MultA(Y);
            assert_bm_orthonormal(Y, BY, m);
            return;
        }
        auto assert_finite = [&]() {
            if(!Y.allFinite()) throw std::runtime_error("block_bm_orthonormalize_eig: Y has nan or inf");
            if(!BY.allFinite()) throw std::runtime_error("block_bm_orthonormalize_eig: BY has nan or inf");
        };

        std::function<MatrixType(const Eigen::Ref<const MatrixType> &)> fMultB = [this](const Eigen::Ref<const MatrixType> &X) -> MatrixType {
            return this->MultB(X);
        };

        Eigen::Index maxReps = 1;
        Eigen::Index rep     = 0;
        for(rep = 0; rep < maxReps; ++rep) {
            assert_finite();

            auto eosm = BmEigOrthoStepMeta<Scalar>(Y, BY);
            {
                auto token_orth_factor = status.time_orth_factor.tic_token();
                do_bm_eig_orthonormalization_step<Scalar, RealScalar, Scalar>(eosm, fMultB, log);
            }

            assert_finite();

            if(eosm.Y.cols() == 0) {
                if(log && log->should_log(spdlog::level::trace)) log->trace("block_bm_orthonormalize_eig: 0/{} cols remain in Y", m.Gram.cols());
                Y  = MatrixType();
                AY = MatrixType();
                BY = MatrixType();
                m  = OrthMeta();
                return;
            }

            {
                auto token_orth_update = status.time_orth_update.tic_token();
                Y                      = eosm.Y.template cast<Scalar>();
                BY                     = eosm.BY.template cast<Scalar>();
            }
            {
                auto token_orth_project = status.time_orth_project.tic_token();
                m.analyze_bm_orthonormality(Y, BY);
            }
            assert_finite();

            if(log && log->should_log(spdlog::level::trace))
                log->trace("block_bm_orthonormalize_eig: eig rep {} | orthError {:.4e} symmError {:.4e} skewError {:.4e} | tol {:.5e}", rep, m.orthError,
                           m.symmError, m.skewError, normTol);
        }
        if(m.skewError >= RealScalar{1e-3f} && log) {
            log->warn("block_bm_orthonormalize_eig: large skew error on rep {} | orthError {:.4e} symmError {:.4e} skewError {:.4e} | cols {}", rep,
                      m.orthError, m.symmError, m.skewError, Y.cols());
        }

        {
            auto token_orth_refresh = status.time_orth_refresh.tic_token();
            AY                      = MultA(Y);
        }
        assert_bm_orthonormal(Y, BY, m);
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::block_bm_orthonormalize(MatrixType &Y, MatrixType &AY, MatrixType &BY, OrthMeta &m) requires(form_ == grit::Form::GENERALIZED)
    {
        auto token_orthonormalize = status.time_orthonormalize.tic_token();
        if(Y.cols() == 0) {
            AY.resizeLike(Y);
            BY.resizeLike(Y);
            return;
        }
        if(m.mask.size() > 0 && m.mask.sum() == 0) return;
        assert(cfg().use_b_inner_product && "block_bm_orthonormalize is for B inner product");

        auto handle_masked_columns = [&]() {
            if(m.mask.sum() == Y.cols()) return;
            switch(m.maskPolicy) {
                case MaskPolicy::COMPRESS:
                    if(log && log->should_log(spdlog::level::debug))
                        log->debug("block_bm_orthonormalize: compressing Y | kept {} of {} | maskTol {:.3e}", m.mask.sum(), Y.cols(), m.maskTol);
                    compress_cols(Y, m.mask);
                    if(BY.rows() == Y.rows() && BY.cols() == m.mask.size()) compress_cols(BY, m.mask);
                    m.mask = VectorIdxT::Ones(Y.cols());
                    if(BY.rows() == Y.rows() && BY.cols() == Y.cols()) m.analyze_bm_orthonormality(Y, BY);
                    break;
                case MaskPolicy::RANDOMIZE:
                    if(log && log->should_log(spdlog::level::debug))
                        log->debug("block_bm_orthonormalize: randomizing masked columns | kept {} of {} | maskTol {:.3e}", m.mask.sum(), Y.cols(), m.maskTol);
                    for(Eigen::Index j = 0; j < Y.cols(); ++j) {
                        if(m.mask(j) == 0) Y.col(j) = Eigen::VectorXf::Random(Y.col(j).size()).template cast<Scalar>();
                    }
                    {
                        auto token_orth_refresh = status.time_orth_refresh.tic_token();
                        BY                      = MultB(Y);
                    }
                    {
                        auto token_orth_project = status.time_orth_project.tic_token();
                        m.analyze_bm_orthonormality(Y, BY);
                    }
                    break;
            }
        };

        m.mask       = VectorIdxT::Ones(Y.cols());
        m.proj_sum_b = VectorReal::Zero(Y.cols());
        m.scale_log  = VectorReal::Zero(Y.cols());
        if(std::isnan(m.maskTol)) m.maskTol = normTol * static_cast<RealScalar>(Y.cols());
        if(std::isnan(m.orthTol)) m.orthTol = normTol * static_cast<RealScalar>(Y.cols());

        if(m.refresh_by || Y.cols() != BY.cols() || Y.rows() != BY.rows()) {
            auto token_orth_refresh = status.time_orth_refresh.tic_token();
            BY                      = MultB(Y);
        }
        m.refresh_by = false;
        {
            auto token_orth_project = status.time_orth_project.tic_token();
            m.analyze_bm_orthonormality(Y, BY);
        }
        assert_allFinite(BY);

        m.Rdiag = VectorReal::Zero(Y.cols());
        for(Eigen::Index j = 0; j < Y.cols(); ++j) {
            auto       yj     = Y.col(j);
            auto       byj    = BY.col(j);
            RealScalar normSq = std::real(yj.dot(byj));
            RealScalar norm   = std::sqrt(std::max<RealScalar>(0, normSq));
            m.Rdiag(j)        = norm;
            if(norm < m.maskTol) {
                if(log && log->should_log(spdlog::level::trace)) log->trace("masking Y col {} | bm norm {:.3e} | maskTol {:.3e}", j, norm, m.maskTol);
                m.mask(j) = 0;
                yj.setZero();
                byj.setZero();
            }
        }
        {
            auto token_orth_mask = status.time_orth_mask.tic_token();
            handle_masked_columns();
        }
        if(Y.cols() == 0) {
            AY.resizeLike(Y);
            BY.resizeLike(Y);
            return;
        }

        Eigen::Index maxReps = 2;
        for(Eigen::Index rep = 0; rep < maxReps; ++rep) {
            VectorReal normSqs = VectorReal::Zero(Y.cols());
            VectorIdxT have    = VectorIdxT::Zero(Y.cols());

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

                    RealScalar normSq;
                    Scalar     proj1;
                    Scalar     proj2;
                    Scalar     proj_ij;
                    {
                        auto token_orth_project = status.time_orth_project.tic_token();
                        normSq                  = normSqs(i);
                        proj1                   = yi.dot(byj);
                        proj2                   = byi.dot(yj);
                        proj_ij                 = normSq > std::numeric_limits<RealScalar>::min() ? (proj1 + proj2) / (RealScalar{2} * normSq) : Scalar{0};
                    }

                    {
                        auto token_orth_update  = status.time_orth_update.tic_token();
                        yj.noalias()           -= yi * proj_ij;
                        byj.noalias()          -= byi * proj_ij;
                    }
                }

                RealScalar normSq = std::real(yj.dot(byj));
                RealScalar norm   = std::sqrt(std::max<RealScalar>(0, normSq));
                if(norm <= m.maskTol) {
                    if(log && log->should_log(spdlog::level::trace)) log->trace("masking Y col {} | bm norm {:.3e} | maskTol {:.3e}", j, norm, m.maskTol);
                    m.mask(j) = 0;
                    yj.setZero();
                    byj.setZero();
                    continue;
                }

                yj         /= norm;
                byj        /= norm;
                normSqs(j)  = std::max<RealScalar>(0, std::real(yj.dot(byj)));
            }

            {
                auto token_orth_project = status.time_orth_project.tic_token();
                m.analyze_bm_orthonormality(Y, BY);
            }
            {
                auto token_orth_mask = status.time_orth_mask.tic_token();
                handle_masked_columns();
            }
            if(Y.cols() == 0) {
                AY.resizeLike(Y);
                BY.resizeLike(Y);
                return;
            }
            if(log && log->should_log(spdlog::level::trace))
                log->trace("block_bm_orthonormalize: dgks rep {} | orthError {:.4e} symmError {:.4e} skewError {:.4e}", rep, m.orthError, m.symmError,
                           m.skewError);
            if(m.orthError < m.orthTol) break;
        }

        {
            auto token_orth_refresh = status.time_orth_refresh.tic_token();
            AY                      = MultA(Y);
        }
        assert_bm_orthonormal(Y, BY, m);
    }
}

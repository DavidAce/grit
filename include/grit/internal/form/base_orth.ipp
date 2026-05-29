#pragma once

#include <grit/form/base.h>

namespace grit::form {
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
    void base<Scalar, form_>::OrthMeta::analyze_b_orthonormality(const Eigen::Ref<const MatrixType> &Y, const Eigen::Ref<const MatrixType> &B_Y) {
        analyze_bm_orthonormality(Y, B_Y);
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::OrthMeta::analyze_bm_orthonormality(const Eigen::Ref<const MatrixType> &Y, const Eigen::Ref<const MatrixType> &B_Y) {
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
    void base<Scalar, form_>::OrthMeta::analyze_bm_orthogonality(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &B_X,
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

        if(orthError > finalTol && eiglog)
            eiglog->warn("{}:{}: {}: matrix is not L2-orthonormal: error {:.5e} > tol {:.5e}", location.file_name(), location.line(), location.function_name(),
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

        if(orthError > finalTol && eiglog)
            eiglog->warn("{}:{}: {}: matrices are not L2-orthogonal: error {:.5e} > tol {:.5e}", location.file_name(), location.line(),
                         location.function_name(), orthError, finalTol);
        if(orthError > RealScalar{1000} * finalTol)
            throw std::runtime_error(fmt::format("{}:{}: {}: matrices are not L2-orthogonal: error {:.5e} > tol {:.5e}", location.file_name(), location.line(),
                                                 location.function_name(), orthError, finalTol));
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::assert_bm_orthonormal(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &B_X, const OrthMeta &m,
                                             const std::source_location &location) {
        if(X.cols() == 0) return;
        if(X.cols() != B_X.cols() || X.rows() != B_X.rows())
            throw std::runtime_error(
                fmt::format("{}:{}: {}: X and B_X have incompatible dimensions", location.file_name(), location.line(), location.function_name()));

        MatrixType G1        = X.adjoint() * B_X;
        MatrixType G2        = B_X.adjoint() * X;
        MatrixType Gram      = G1;
        MatrixType Gram_symm = (G1 + G2) * half;
        MatrixType Gram_skew = (G1 - G2) * half;
        MatrixType I         = MatrixType::Identity(Gram.rows(), Gram.cols());
        RealScalar orthError = (Gram - I).norm();
        RealScalar symmError = (Gram_symm - I).norm();
        RealScalar skewError = Gram_skew.norm();

        RealScalar xnorm    = X.norm();
        RealScalar bxnorm   = B_X.norm();
        RealScalar bnorm    = std::isfinite(status.op_norm_estimate) ? status.op_norm_estimate : RealScalar{1};
        RealScalar t_abs    = orthTol * static_cast<RealScalar>(X.cols()) * (xnorm + bxnorm);
        RealScalar bmTol    = orthTol * static_cast<RealScalar>(X.cols()) * bnorm;
        RealScalar maskTol  = std::isfinite(m.maskTol) ? m.maskTol : orthTol;
        RealScalar finalTol = std::max({t_abs, orthTol, bmTol, maskTol}) * RealScalar{10};

        RealScalar error = std::max({orthError, symmError, skewError});
        if(error > finalTol && eiglog)
            eiglog->warn("{}:{}: {}: matrix is not B-orthonormal: error {:.5e} > tol {:.5e} | orth {:.5e} symm {:.5e} skew {:.5e}", location.file_name(),
                         location.line(), location.function_name(), error, finalTol, orthError, symmError, skewError);
        if(error > RealScalar{1000} * finalTol)
            throw std::runtime_error(fmt::format("{}:{}: {}: matrix is not B-orthonormal: error {:.5e} > tol {:.5e}", location.file_name(), location.line(),
                                                 location.function_name(), error, finalTol));
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::assert_bm_orthogonal(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &B_Y, const OrthMeta &m,
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
            eiglog->warn("{}:{}: {}: matrices are not B-orthogonal: error {:.5e} > tol {:.5e}", location.file_name(), location.line(), location.function_name(),
                         orthError, finalTol);
        if(orthError > RealScalar{1000} * finalTol)
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
    void base<Scalar, form_>::block_l2_orthogonalize(const MatrixType &X, const MatrixType &AX, MatrixType &Y, MatrixType &AY, OrthMeta &m) {
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
        if constexpr(settings::debug_ortho) {
            if(eiglog && eiglog->should_log(spdlog::level::trace))
                eiglog->trace("rep {} orthError after l2 orthogonalization: {:.3e} | orthTol {:.3e}", rep, m.orthError, m.orthTol);
        }
        assert_l2_orthogonal(X, Y, m);
        AY = MultA(Y);
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::block_l2_orthogonalize(const MatrixType &X, const MatrixType &AX, const MatrixType &BX, MatrixType &Y, MatrixType &AY, MatrixType &BY,
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
        if constexpr(form_ == grit::Form::GENERALIZED)
            BY = MultB(Y);
        else
            BY = Y;
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::block_bm_orthogonalize(const MatrixType &X, const MatrixType &AX, const MatrixType &BX, MatrixType &Y, MatrixType &AY,
                                                     MatrixType &BY, OrthMeta &m) requires(form_ == grit::Form::GENERALIZED) {
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
                eiglog->trace("block_bm_orthogonalize: no need: orthError {:.4e} symmError {:.4e} skewError {:.4e} Eyy {:.4e} orthTol {:.4e}", m.orthError,
                              m.symmError, m.skewError, Eyy, m.orthTol);
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
            eiglog->trace("rep {} orthError after bm orthogonalization: {:.3e} | symm {:.3e} | skew {:.3e} | orthTol {:.3e}", rep, m.orthError, m.symmError,
                          m.skewError, m.orthTol);

        AY = MultA(Y);
        assert_bm_orthogonal(X, BY, m);
        assert_bm_orthogonal(BX, Y, m);
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::block_l2_orthonormalize(MatrixType &Y, MatrixType &AY, OrthMeta &m) {
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
                if(eiglog && eiglog->should_log(spdlog::level::trace)) eiglog->trace("masking Y col {} | norm {:.3e} | maskTol {:.3e}", j, norm, m.maskTol);
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
                if(eiglog && eiglog->should_log(spdlog::level::trace)) eiglog->trace("masking Y col {} | norm {:.3e} | maskTol {:.3e}", j, norm, m.maskTol);
                m.mask(j) = 0;
                yj.setZero();
            }
        }
        handle_masked_columns();

        AY = MultA(Y);
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
        if constexpr(form_ == grit::Form::GENERALIZED)
            BY = MultB(Y);
        else
            BY = Y;
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::block_bm_orthonormalize(MatrixType &Y, MatrixType &AY, MatrixType &BY, OrthMeta &m) requires(form_ == grit::Form::GENERALIZED) {
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

        m.mask       = VectorIdxT::Ones(Y.cols());
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
                if(eiglog && eiglog->should_log(spdlog::level::trace)) eiglog->trace("masking Y col {} | bm norm {:.3e} | maskTol {:.3e}", j, norm, m.maskTol);
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

                yj         /= norm;
                byj        /= norm;
                normSqs(j)  = std::max<RealScalar>(0, std::real(yj.dot(byj)));
            }

            m.analyze_bm_orthonormality(Y, BY);
            handle_masked_columns();
            if(Y.cols() == 0) {
                AY.resizeLike(Y);
                BY.resizeLike(Y);
                return;
            }
            if(eiglog && eiglog->should_log(spdlog::level::trace))
                eiglog->trace("block_bm_orthonormalize: dgks rep {} | orthError {:.4e} symmError {:.4e} skewError {:.4e}", rep, m.orthError, m.symmError,
                              m.skewError);
            if(m.orthError < m.orthTol) break;
        }

        AY = MultA(Y);
        assert_bm_orthonormal(Y, BY, m);
    }
}

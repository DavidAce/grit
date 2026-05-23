#pragma once

#include <grit/form/base.h>
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
        A.set_iterativeLinearSolverConfig(1000, RealScalar{0.25f}, MatDef::IND);
        A.set_jcbMaxBlockSize(Eigen::Index{-1});
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
    void base<Scalar>::set_form_jcbMaxBlockSize([[maybe_unused]] Eigen::Index jcbMaxBlockSize) {}
    template<typename Scalar>
    void base<Scalar>::set_form_jcbOverlapSize([[maybe_unused]] Eigen::Index jcbOverlapSize) {}
    template<typename Scalar>
    void base<Scalar>::set_form_jcbNumPasses([[maybe_unused]] Eigen::Index jcbNumPasses) {}
    template<typename Scalar>
    void base<Scalar>::set_form_preconditioner_type([[maybe_unused]] Preconditioner preconditioner_type) {}
    template<typename Scalar>
    void base<Scalar>::set_form_preconditioner_params([[maybe_unused]] Eigen::Index maxiters, [[maybe_unused]] RealScalar initialTol,
                                                      [[maybe_unused]] Eigen::Index jcbMaxBlockSize) {}

    template<typename Scalar>
    void base<Scalar>::set_jcbMaxBlockSize(Eigen::Index jcbMaxBlockSize) {
        A.set_jcbMaxBlockSize(jcbMaxBlockSize);
        set_form_jcbMaxBlockSize(jcbMaxBlockSize);
    }
    template<typename Scalar>
    void base<Scalar>::set_jcbOverlapSize(Eigen::Index jcbOverlapSize) {
        A.set_jcbOverlapSize(jcbOverlapSize);
        set_form_jcbOverlapSize(jcbOverlapSize);
    }
    template<typename Scalar>
    void base<Scalar>::set_jcbNumPasses(Eigen::Index jcbNumPasses) {
        A.set_jcbNumPasses(jcbNumPasses);
        set_form_jcbNumPasses(jcbNumPasses);
    }
    template<typename Scalar>
    Eigen::Index base<Scalar>::get_jcbMaxBlockSize() const {
        return A.get_jcbMaxBlockSize();
    }
    template<typename Scalar>
    Eigen::Index base<Scalar>::get_jcbOverlapSize() const {
        return A.get_jcbOverlapSize();
    }
    template<typename Scalar>
    Eigen::Index base<Scalar>::get_jcbNumPasses() const {
        return A.get_jcbNumPasses();
    }

    template<typename Scalar>
    void base<Scalar>::set_preconditioner_type(Preconditioner preconditioner_type_) {
        preconditioner_type = preconditioner_type_;
        A.preconditioner    = preconditioner_type;
        set_form_preconditioner_type(preconditioner_type);
        use_preconditioner = preconditioner_type != Preconditioner::NONE;
    }

    template<typename Scalar>
    void base<Scalar>::set_preconditioner_params(Eigen::Index maxiters, RealScalar initialTol, Eigen::Index jcbMaxBlockSize) {
        A.set_iterativeLinearSolverConfig(maxiters, initialTol, MatDef::IND);
        A.set_jcbMaxBlockSize(jcbMaxBlockSize);
        set_form_preconditioner_params(maxiters, initialTol, jcbMaxBlockSize);
    }

    template<typename Scalar>
    typename base<Scalar>::MatrixType base<Scalar>::MultA(const Eigen::Ref<const MatrixType> &X) {
        auto token_matvecs  = status.time_matvecs.tic_token();
        status.num_matvecs += X.cols();
        return A.MultAX(X);
    }

    template<typename Scalar>
    typename base<Scalar>::MatrixType base<Scalar>::MultP(const Eigen::Ref<const MatrixType> &X, [[maybe_unused]] const Eigen::Ref<const VectorReal> &evals,
                                                          std::optional<const Eigen::Ref<const MatrixType>> initialGuess) {
        auto token_precond                                = status.time_precond.tic_token();
        A.get_iterativeLinearSolverConfig().initialGuess  = initialGuess.value_or(MatrixType{});
        MatrixType PX                                     = A.MultPX(X, evals);
        status.num_precond                               += X.cols();
        return PX;
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
        if(V.rows() != N || V.cols() == 0) V = MatrixType::Random(N, std::max<Eigen::Index>(nev, b));
        OrthMeta   m;
        MatrixType AV_init = MultA(V);
        block_l2_orthonormalize(V, AV_init, m);
        Q                  = V.leftCols(std::min<Eigen::Index>(V.cols(), ncv));
        AQ                 = MultA(Q);
        BQ                 = MultB(Q);
        status.rNorms_init = VectorReal::Constant(nev, std::numeric_limits<RealScalar>::infinity());
    }

    template<typename Scalar>
    void base<Scalar>::extractRitzVectors() {
        status.optIdx = get_ritz_indices(ritz, 0, nev, T_evals);
        auto Z        = T_evecs(Eigen::placeholders::all, status.optIdx);
        status.eigVal.resize(nev);
        for(Eigen::Index i = 0; i < nev; ++i) status.eigVal(i) = T_evals(status.optIdx[static_cast<size_t>(i)]);
        V                    = Q * Z;
        AV                   = AQ * Z;
        BV                   = BQ * Z;
        auto [S_out, rNorms] = get_residuals(status.eigVal, AV, BV);
        S                    = S_out;
        status.rNorms        = rNorms;
        if(status.rNorms_init.size() != status.rNorms.size()) status.rNorms_init = status.rNorms;
    }

    template<typename Scalar>
    void base<Scalar>::preamble() {
        status.num_matvecs = 0;
        status.num_precond = 0;
    }

    template<typename Scalar>
    void base<Scalar>::updateStatus() {
        status.num_matvecs_total += status.num_matvecs;
        status.num_precond_total += status.num_precond;
        if((status.rNorms.array() < rNormTols().array()).all()) status.stopReason = StopReason::converged;
        if(max_iters >= 0 && status.iter >= max_iters) status.stopReason = StopReason::max_iters;
        if(max_matvecs >= 0 && status.num_matvecs_total >= max_matvecs) status.stopReason = StopReason::max_matvecs;
    }

    template<typename Scalar>
    void base<Scalar>::printStatus() {
        if(eiglog && eiglog->should_log(spdlog::level::debug) && status.eigVal.size() > 0) {
            eiglog->debug("iter {} | eval {:.16e} | rnorm {:.3e}", status.iter, status.eigVal(0), status.rNorms(0));
        }
    }

    template<typename Scalar>
    result_view<Scalar> base<Scalar>::result() const {
        return result_view<Scalar>(*this);
    }

    template<typename Scalar>
    void base<Scalar>::clear_result() {
        status = Status{};
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
        if(!X.allFinite()) throw std::runtime_error(std::string("non-finite matrix at ") + location.function_name());
    }

    template<typename Scalar>
    void base<Scalar>::block_l2_orthonormalize(MatrixType &Y, MatrixType &AY, OrthMeta &m) {
        Eigen::HouseholderQR<MatrixType> qr(Y);
        auto                             cols = std::min(Y.rows(), Y.cols());
        MatrixType                       Qtmp = qr.householderQ() * MatrixType::Identity(Y.rows(), cols);
        Y                                     = Qtmp;
        AY                                    = MultA(Y);
        m.analyze_l2_orthonormality(Y);
    }

    template<typename Scalar>
    void base<Scalar>::block_l2_orthonormalize(MatrixType &Y, MatrixType &AY, MatrixType &BY, OrthMeta &m) {
        block_l2_orthonormalize(Y, AY, m);
        BY = MultB(Y);
    }
}

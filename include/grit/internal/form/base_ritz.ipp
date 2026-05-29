#pragma once

#include <grit/form/base.h>

namespace grit::form {
    template<typename Scalar, grit::Form form_>
    std::vector<Eigen::Index> base<Scalar, form_>::get_ritz_indices(OptRitz ritz, Eigen::Index offset, Eigen::Index num, const VectorReal &evals) const {
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

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::extractRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &AV, MatrixType &S, VectorReal &rNorms) {
        MatrixType Z = T_evecs(Eigen::placeholders::all, optIdx);
        VectorReal Y = T_evals(optIdx);

        V  = Q * Z;
        AV = AQ * Z;
        S  = get_residuals(Y, AV, V, rNorms);
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::extractRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &AV, MatrixType &BV, MatrixType &S,
                                          VectorReal &rNorms) {
        MatrixType Z = T_evecs(Eigen::placeholders::all, optIdx);
        VectorReal Y = T_evals(optIdx);

        V.noalias()  = Q * Z;
        AV.noalias() = AQ * Z;
        BV.noalias() = BQ * Z;
        S            = get_residuals(Y, AV, BV, rNorms);
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::extractRitzVectors() {
        if(status.stopReason != StopReason::none) return;
        if(T_evals.size() < cfg().block_size) return;

        Eigen::Index k     = std::min(cfg().block_size, T_evals.size());
        Eigen::Index nritz = std::max({cfg().nev, cfg().block_size, k});

        status.optIdx = get_ritz_indices(cfg().ritz, 0, nritz, T_evals);

        if(cfg().use_refined_rayleigh_ritz) {
            if constexpr(form_ == grit::Form::GENERALIZED) {
                refinedRitzVectors(status.optIdx, V, AV, BV, S, status.rNorms);
            } else {
                refinedRitzVectors(status.optIdx, V, AV, S, status.rNorms);
                BV = V;
            }
        } else {
            if constexpr(form_ == grit::Form::GENERALIZED) {
                extractRitzVectors(status.optIdx, V, AV, BV, S, status.rNorms);
            } else {
                extractRitzVectors(status.optIdx, V, AV, S, status.rNorms);
                BV = V;
            }
        }

        K_prev = K;
        K      = V.leftCols(k);

        if(k > cfg().block_size) {
            V.conservativeResize(Eigen::NoChange, cfg().block_size);
            AV.conservativeResize(Eigen::NoChange, cfg().block_size);
            BV.conservativeResize(Eigen::NoChange, cfg().block_size);
            S.conservativeResize(Eigen::NoChange, cfg().block_size);
            status.rNorms.conservativeResize(cfg().block_size);
        }

        if(status.rNorms_init.size() != status.rNorms.size()) status.rNorms_init = status.rNorms;
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::orthonormalize_Z(Eigen::Ref<MatrixType> Z, const Eigen::Ref<const MatrixType> &T2) {
        if(!cfg().use_b_inner_product) {
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

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::MatrixType base<Scalar, form_>::get_refined_ritz_eigenvectors_std(const Eigen::Ref<const MatrixType> &Z,
                                                                                      const Eigen::Ref<const VectorReal> &Y, const MatrixType &Q,
                                                                                      const MatrixType &AQ) {
        assert(!cfg().use_b_inner_product);
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

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::MatrixType base<Scalar, form_>::get_refined_ritz_eigenvectors_gen(const Eigen::Ref<const MatrixType> &Z,
                                                                                                      const Eigen::Ref<const VectorReal> &Y,
                                                                                                      const MatrixType &AQ,
                                                                                                      const MatrixType &BQ) requires(form_ == grit::Form::GENERALIZED) {
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

                if(cfg().use_b_inner_product) {
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

    template<typename Scalar, grit::Form form_>
    std::pair<typename base<Scalar, form_>::MatrixType, typename base<Scalar, form_>::MatrixType>
        base<Scalar, form_>::get_bm_normalizer_for_the_projected_pencil(const MatrixType &T2) {
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

    template<typename Scalar, grit::Form form_>
    typename base<Scalar, form_>::MatrixType base<Scalar, form_>::get_optimal_rayleigh_ritz_matrix(const MatrixType &Z_rr, const MatrixType &Z_ref, const MatrixType &T1,
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
                auto select1 = get_ritz_indices(cfg().ritz, 0, 1, ges.eigenvalues());
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

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::refinedRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &AQ, MatrixType &BQ, MatrixType &S,
                                                 VectorReal &rNorms) requires(form_ == grit::Form::GENERALIZED) {
        VectorReal Y     = T_evals(optIdx);
        MatrixType Z_rr  = T_evecs(Eigen::placeholders::all, optIdx);
        MatrixType Z_ref = get_refined_ritz_eigenvectors_gen(Z_rr, Y, AQ, BQ);
        MatrixType Z_opt = get_optimal_rayleigh_ritz_matrix(Z_rr, Z_ref, T1, T2);

        V.noalias()  = Q * Z_opt;
        AQ.noalias() = this->AQ * Z_opt;
        BQ.noalias() = this->BQ * Z_opt;

        if(cfg().use_rayleigh_quotients_instead_of_evals) {
            VectorReal rq1  = (V.adjoint() * AQ).diagonal().real();
            VectorReal rq2  = (V.adjoint() * BQ).diagonal().real();
            T_evals(optIdx) = rq1.cwiseQuotient(rq2);
            Y               = T_evals(optIdx);
        }

        S = get_residuals(Y, AQ, BQ, rNorms);
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::refinedRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &AQ, MatrixType &S, VectorReal &rNorms) {
        MatrixType Z     = T_evecs(Eigen::placeholders::all, optIdx);
        VectorReal Y     = T_evals(optIdx);
        MatrixType Z_ref = get_refined_ritz_eigenvectors_std(Z, Y, Q, this->AQ);

        V  = Q * Z_ref;
        AQ = this->AQ * Z_ref;

        if(cfg().use_rayleigh_quotients_instead_of_evals) { Y = (V.adjoint() * AQ).diagonal().real(); }

        S = get_residuals(Y, AQ, V, rNorms);
    }

    template<typename Scalar, grit::Form form_>
    void base<Scalar, form_>::refinedRitzVectors() {
        if(!cfg().use_refined_rayleigh_ritz) return;
        if(status.rNorms.size() == 0) throw std::runtime_error("refineRitzVectors() called before extractRitzVectors()");
        if constexpr(form_ == grit::Form::GENERALIZED) {
            refinedRitzVectors(status.optIdx, V, AV, BV, S, status.rNorms);
        } else {
            refinedRitzVectors(status.optIdx, V, AV, S, status.rNorms);
        }
    }

}

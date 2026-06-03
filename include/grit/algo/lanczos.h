#pragma once

#include <Eigen/Core>
#include <functional>
#include <grit/form/base.h>
#include <limits>
#include <optional>
#include <spdlog/common.h>
#include <type_traits>

namespace grit::algo {
    template<typename Scalar_, grit::Form form_>
    class lanczos : public form::base<Scalar_, form_> {
        public:
        using Base       = form::base<Scalar_, form_>;
        using Scalar     = typename Base::Scalar;
        using RealScalar = typename Base::RealScalar;
        using MatrixType = typename Base::MatrixType;
        using VectorType = typename Base::VectorType;
        using VectorReal = typename Base::VectorReal;
        using VectorIdxT = typename Base::VectorIdxT;
        using OrthMeta   = typename Base::OrthMeta;
        using BaseConfig = typename Base::BaseConfig;

        using Base::A;
        using Base::AQ;
        using Base::AV;
        using Base::BQ;
        using Base::BV;
        using Base::D;
        using Base::K;
        using Base::K_prev;
        using Base::Q;
        using Base::S;
        using Base::T;
        using Base::T_evals;
        using Base::T_evecs;
        using Base::V;
        using Base::V_prev;
        using Base::W;

        using Base::block_bm_orthogonalize;
        using Base::block_bm_orthonormalize;
        using Base::block_l2_orthogonalize;
        using Base::block_l2_orthonormalize;
        using Base::cfg;
        using Base::eps;
        using Base::get_bm_normalizer_for_the_projected_pencil;
        using Base::get_optimal_rayleigh_ritz_matrix;
        using Base::get_refined_ritz_eigenvectors_gen;
        using Base::get_refined_ritz_eigenvectors_std;
        using Base::hhqr;
        using Base::log;
        using Base::MultA;
        using Base::MultB;
        using Base::MultP;
        using Base::N;
        using Base::normTol;
        using Base::orthonormalize_Z;
        using Base::qBlocks;
        using Base::status;

        struct Config : BaseConfig {
            static constexpr RealScalar                         eps = std::numeric_limits<RealScalar>::epsilon();
            Eigen::Index                                        maxRetainBlocks = 2;
            std::function<void(const lanczos<Scalar, form_> &)> user_callback;
        };

        Config config;

        lanczos(Matvec<Scalar> &A) requires(form_ == grit::Form::STANDARD);
        lanczos(Matvec<Scalar> &A, Matvec<Scalar> &B) requires(form_ == grit::Form::GENERALIZED);

        void run();

        private:
        MatrixType A_block, B_block;
        MatrixType AK, BK;
        bool       beta_stalled = false;

        void write_Q_next_B_DGKS(Eigen::Index i);
        void assert_config() const;
        void assert_operator_config() const requires(form_ == grit::Form::STANDARD);
        void assert_operator_config() const requires(form_ == grit::Form::GENERALIZED);
        void extractRitzVectors() final;
        void updateStatus() final;
        void run_user_callback() final;

        public:
        void build() final;
    };
}

#include <grit/internal/algo/lanczos.ipp>

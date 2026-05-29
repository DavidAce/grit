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
    class lobpcg : public form::base<Scalar_, form_> {
        public:
        using Base                   = form::base<Scalar_, form_>;
        using Scalar                 = typename Base::Scalar;
        using RealScalar             = typename Base::RealScalar;
        using MatrixType             = typename Base::MatrixType;
        using VectorType             = typename Base::VectorType;
        using VectorReal             = typename Base::VectorReal;
        using VectorIdxT             = typename Base::VectorIdxT;
        using BaseConfig             = typename Base::BaseConfig;

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
        using Base::T1;
        using Base::T2;
        using Base::T_evals;
        using Base::T_evecs;
        using Base::V;
        using Base::V_prev;
        using Base::W;

        using Base::MultA;
        using Base::MultB;
        using Base::MultP;
        using Base::N;
        using Base::assert_allFinite;
        using Base::block_bm_orthogonalize;
        using Base::block_bm_orthonormalize;
        using Base::block_l2_orthogonalize;
        using Base::block_l2_orthonormalize;
        using Base::cfg;
        using Base::clear_result;
        using Base::eiglog;
        using Base::eps;
        using Base::hhqr;
        using Base::qBlocks;
        using Base::status;

        struct Config : BaseConfig {
            static constexpr RealScalar                        eps = std::numeric_limits<RealScalar>::epsilon();
            bool                                               inject_randomness = false;
            std::function<void(const lobpcg<Scalar, form_> &)> user_callback;
            Eigen::Index                                       max_basis_blocks         = 8;
            Eigen::Index                                       max_extra_ritz_history   = 1;
            Eigen::Index                                       max_ritz_residual_history = 1;
        };

        Config config;

        lobpcg(Matvec<Scalar> &A) requires(form_ == grit::Form::STANDARD);
        lobpcg(Matvec<Scalar> &A, Matvec<Scalar> &B) requires(form_ == grit::Form::GENERALIZED);

        void               set_initial_guess(const MatrixType &V);
        void               clear_initial_guess();
        [[nodiscard]] bool has_initial_guess() const;
        void               run();

        private:
        Eigen::Index      max_wBlocks       = 1;
        Eigen::Index      max_mBlocks       = 1;
        Eigen::Index      max_sBlocks       = 1;
        Eigen::Index      wBlocks           = 0;
        Eigen::Index      mBlocks           = 0;
        Eigen::Index      rBlocks           = 0;
        Eigen::Index      sBlocks           = 0;
        MatrixType        G;
        const MatrixType *initial_guess_ptr = nullptr;
        MatrixType        empty_initial_guess;

        void shift_blocks_right(Eigen::Ref<MatrixType> matrix, Eigen::Index offset_old, Eigen::Index offset_new, Eigen::Index extent);
        void roll_blocks_left(Eigen::Ref<MatrixType> matrix, Eigen::Index offset, Eigen::Index extent);
        std::pair<VectorIdxT, VectorIdxT> selective_orthonormalize();
        void                              assert_config() const;
        [[nodiscard]] const MatrixType        &initial_guess() const;
        [[nodiscard]] static const MatrixType &default_initial_guess();
        void                                   assert_operator_config() const requires(form_ == grit::Form::STANDARD);
        void                                   assert_operator_config() const requires(form_ == grit::Form::GENERALIZED);
        void                                   extractRitzVectors() final;
        void                                   run_user_callback() final;

        public:
        void build() final;
    };
}

#pragma once

#include <Eigen/Core>
#include <functional>
#include "grit/form/base.h"
#include <limits>
#include <optional>
#include <spdlog/common.h>
#include <type_traits>

namespace grit::algo {
    /*! Block Lanczos solver for symmetric or Hermitian eigenvalue problems. */
    template<typename Scalar_, grit::Form form_>
    class lanczos : public form::base<Scalar_, form_> {
        public:
        using Base       = form::base<Scalar_, form_>; /*!< Base form for the selected problem type. */
        using Scalar     = typename Base::Scalar;      /*!< Scalar type of vectors and operators. */
        using RealScalar = typename Base::RealScalar;  /*!< Real scalar type used for Ritz values and norms. */
        using MatrixType = typename Base::MatrixType;  /*!< Dense block of vectors. */
        using VectorType = typename Base::VectorType;  /*!< Dense single vector. */
        using VectorReal = typename Base::VectorReal;  /*!< Real-valued vector. */
        using VectorIdxT = typename Base::VectorIdxT;  /*!< Vector of column indices. */
        using OrthMeta   = typename Base::OrthMeta;    /*!< Orthogonalization diagnostics. */
        using BaseConfig = typename Base::BaseConfig;  /*!< Shared solver configuration. */

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

        /*! Configuration for block Lanczos. */
        struct Config : BaseConfig {
            static constexpr RealScalar                         eps             = std::numeric_limits<RealScalar>::epsilon(); /*!< Machine epsilon. */
            Eigen::Index                                        maxRetainBlocks = 2; /*!< Number of old Lanczos blocks kept on restart. */
            std::function<void(const lanczos<Scalar, form_> &)> user_callback;       /*!< Callback called after each outer iteration. */
        };

        Config config; /*!< User-facing Lanczos configuration. */

        /*!
         * Construct a standard Lanczos solver.
         * @param A Matrix-free operator A.
         */
        lanczos(Matvec<Scalar> &A) requires(form_ == grit::Form::STANDARD);
        /*!
         * Construct a generalized Lanczos solver.
         * @param A Matrix-free operator A.
         * @param B Matrix-free operator B.
         */
        lanczos(Matvec<Scalar> &A, Matvec<Scalar> &B) requires(form_ == grit::Form::GENERALIZED);

        /*! Run the solver until convergence or a stop condition is reached. */
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
        /*! Expand or restart the Lanczos search space. */
        void build() final;
    };
}

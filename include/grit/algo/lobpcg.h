#pragma once

#include "grit/form/base.h"
#include <Eigen/Core>
#include <functional>
#include <limits>
#include <optional>
#include <spdlog/common.h>
#include <type_traits>

namespace grit::algo {
    /*! LOBPCG solver for symmetric or Hermitian eigenvalue problems. */
    template<typename Scalar_, grit::Form form_> class lobpcg : public form::base<Scalar_, form_> {
        public:
        using Base       = form::base<Scalar_, form_>; /*!< Base form for the selected problem type. */
        using Scalar     = typename Base::Scalar;      /*!< Scalar type of vectors and operators. */
        using RealScalar = typename Base::RealScalar;  /*!< Real scalar type used for Ritz values and norms. */
        using MatrixType = typename Base::MatrixType;  /*!< Dense block of vectors. */
        using VectorType = typename Base::VectorType;  /*!< Dense single vector. */
        using VectorReal = typename Base::VectorReal;  /*!< Real-valued vector. */
        using VectorIdxT = typename Base::VectorIdxT;  /*!< Vector of column indices. */
        using BaseConfig = typename Base::BaseConfig;  /*!< Shared solver configuration. */

        using Base::A;
        using Base::AQ;
        using Base::AV;
        using Base::BQ;
        using Base::BV;
        using Base::K;
        using Base::Q;
        using Base::S;
        using Base::T1;
        using Base::T2;
        using Base::T_evals;
        using Base::T_evecs;
        using Base::V;
        using Base::W;

        using Base::assert_allFinite;
        using Base::block_bm_orthogonalize;
        using Base::block_bm_orthonormalize;
        using Base::block_l2_orthogonalize;
        using Base::block_l2_orthonormalize;
        using Base::MultA;
        using Base::MultB;
        using Base::MultP;
        using Base::N;
        using Base::status;

        /*! Configuration for LOBPCG. */
        struct Config : BaseConfig {
            bool                                               inject_randomness = false; /*!< Randomize dependent correction vectors. */
            std::function<void(const lobpcg<Scalar, form_> &)> user_callback;             /*!< Callback called after each outer iteration. */
        };

        Config config; /*!< User-facing LOBPCG configuration. */

        /*!
         * Construct a standard LOBPCG solver.
         * @param A Matrix-free operator A.
         */
        lobpcg(Matvec<Scalar> &A) requires(form_ == grit::Form::STANDARD);
        /*!
         * Construct a generalized LOBPCG solver.
         * @param A Matrix-free operator A.
         * @param B Matrix-free operator B.
         */
        lobpcg(Matvec<Scalar> &A, Matvec<Scalar> &B) requires(form_ == grit::Form::GENERALIZED);

        /*! Run the solver until convergence or a stop condition is reached. */
        void run();

        private:
        Eigen::Index x_blocks = 1;
        Eigen::Index w_blocks = 0;
        Eigen::Index p_blocks = 0;
        MatrixType   AW, BW;
        MatrixType   P, AP, BP;

        [[nodiscard]] const BaseConfig &cfg() const final { return config; }
        void                            assert_config() const;
        void                            extractRitzVectors() final;
        void                            run_user_callback() final;

        public:
        /*! Expand or restart the LOBPCG search space. */
        void build() final;
    };
}

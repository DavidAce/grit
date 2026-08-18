#pragma once

#include "grit/form/base.h"
#include <Eigen/Core>
#include <functional>
#include <limits>
#include <optional>
#include <spdlog/common.h>
#include <type_traits>

namespace grit::algo {
    /*! Generalized Davidson plus Krylov correction solver. */
    template<typename Scalar_, grit::Form form_> class gdplusk : public form::base<Scalar_, form_> {
        public:
        using Base                   = form::base<Scalar_, form_>;            /*!< Base form for the selected problem type. */
        using Scalar                 = typename Base::Scalar;                 /*!< Scalar type of vectors and operators. */
        using RealScalar             = typename Base::RealScalar;             /*!< Real scalar type used for Ritz values and norms. */
        using MatrixType             = typename Base::MatrixType;             /*!< Dense block of vectors. */
        using VectorType             = typename Base::VectorType;             /*!< Dense single vector. */
        using VectorReal             = typename Base::VectorReal;             /*!< Real-valued vector. */
        using VectorIdxT             = typename Base::VectorIdxT;             /*!< Vector of column indices. */
        using fMultP_t               = typename Base::fMultP_t;               /*!< Preconditioner callback type. */
        using OrthMeta               = typename Base::OrthMeta;               /*!< Orthogonalization diagnostics. */
        using BaseConfig             = typename Base::BaseConfig;             /*!< Shared solver configuration. */
        using ResidualCorrectionType = typename Base::ResidualCorrectionType; /*!< Residual correction selector. */

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
        using Base::V;

        using Base::MultA;
        using Base::MultA_inner;
        using Base::MultB;
        using Base::MultB_inner;
        using Base::MultP;
        using Base::N;
        using Base::orthonormalize_Z;
        using Base::residual_correction_type_internal;
        using Base::rNormScales;
        using Base::rNormsRel;
        using Base::status;
        using Base::T1;
        using Base::T2;
        using Base::T_evals;
        using Base::T_evecs;

        using Base::assert_allFinite;
        using Base::assert_bm_orthonormal;
        using Base::auto_residual_correction;
        using Base::block_bm_orthogonalize;
        using Base::block_bm_orthonormalize;
        using Base::block_bm_orthonormalize_eig;
        using Base::block_l2_orthogonalize;
        using Base::block_l2_orthonormalize;
        using Base::eps;
        using Base::get_bm_normalizer_for_the_projected_pencil;
        using Base::get_op_norm_estimate;
        using Base::get_optimal_rayleigh_ritz_matrix;
        using Base::get_refined_ritz_eigenvectors_gen;
        using Base::get_refined_ritz_eigenvectors_std;
        using Base::get_residuals;
        using Base::get_ritz_slopes;
        using Base::get_rnorm_slopes;
        using Base::get_standard_deviations;
        using Base::log;
        using Base::normTol;

        /*! Configuration for GD+K. */
        struct Config : BaseConfig {
            ResidualCorrectionType residual_correction_type         = ResidualCorrectionType::NONE; /*!< Correction used to expand the search space. */
            bool                   use_adaptive_inner_tolerance     = false; /*!< Tighten the inner correction tolerance as residuals improve. */
            bool                   use_jd_b_only                    = false; /*!< Use only B in the generalized Jacobi-Davidson projector. */
            bool                   use_krylov_schur_gdplusk_restart = false; /*!< Use Krylov-Schur style restart for GD+K. */
            bool                   inject_randomness                = false; /*!< Randomize dependent correction vectors. */
            Eigen::Index           maxRetainBlocks                  = 1;     /*!< Number of Ritz blocks kept during restart compression. */
            Eigen::Index           maxPrevBlocks                    = 1;     /*!< Number of previous active Ritz blocks kept between outer iterations. */
            RealScalar             inner_tol                        = RealScalar{0.1f}; /*!< Initial tolerance for inner correction solves. */
            Eigen::Index           inner_max_iters                  = 1000;             /*!< Maximum inner iterations in each inner correction solve. */
            Eigen::Index           auto_probe_length                = 5; /*!< Minimum outer iterations using the method tested by each AUTO probe. */
            Eigen::Index           auto_max_probes = 1; /*!< Maximum AUTO probes while the Ritz values remain stabilized; -1 allows unlimited probes. */
            std::function<void(const gdplusk<Scalar, form_> &)> user_callback; /*!< Callback called after each outer iteration. */
        };

        Config config; /*!< User-facing GD+K configuration. */
        /*!
         * Construct a standard GD+K solver.
         * @param A Matrix-free operator A.
         */
        gdplusk(Matvec<Scalar> &A) requires(form_ == grit::Form::STANDARD);
        /*!
         * Construct a generalized GD+K solver.
         * @param A Matrix-free operator A.
         * @param B Matrix-free operator B.
         */
        gdplusk(Matvec<Scalar> &A, Matvec<Scalar> &B) requires(form_ == grit::Form::GENERALIZED);

        /*! Run the solver until convergence or a stop condition is reached. */
        void run();

        private:
        RealScalar                      current_inner_tol = RealScalar{0.1f};
        MatrixType                      Q_new, AQ_new, BQ_new;
        void                            make_new_Q_block();
        [[nodiscard]] const BaseConfig &cfg() const final { return config; }
        void                            assert_config() const;
        void                            preamble() final;
        void                            extractRitzVectors() final;
        void                            run_user_callback() final;

        public:
        /*!
         * Update the inner correction tolerance from the current search block.
         * @param S Residual or correction block used to estimate current progress.
         */
        void adjust_inner_tolerance(const Eigen::Ref<const MatrixType> &S);
        /*! Update the active residual correction method. */
        void adjust_residual_correction_type();
        /*! Update AUTO residual correction counters after an outer iteration. */
        void update_auto_residual_correction_state();

        /*!
         * Compute the cheap Olsen correction block.
         * @param V Current Ritz vectors.
         * @param S Current residual block.
         * @return Correction block.
         */
        [[nodiscard]] MatrixType cheap_Olsen_correction(const MatrixType &V, const MatrixType &S);
        /*!
         * Compute the cheap Olsen correction block for selected Ritz pairs.
         * @param V Selected Ritz vectors.
         * @param BV B applied to the selected Ritz vectors.
         * @param S Selected residual block.
         * @return Correction block.
         */
        [[nodiscard]] MatrixType cheap_Olsen_correction(const MatrixType &V, const MatrixType &BV, const MatrixType &S);

        /*!
         * Compute the full Olsen correction block.
         * @param V Current Ritz vectors.
         * @param S Current residual block.
         * @return Correction block.
         */
        [[nodiscard]] MatrixType full_Olsen_correction(const MatrixType &V, const MatrixType &S);
        /*!
         * Compute the full Olsen correction block for selected Ritz pairs.
         * @param V Selected Ritz vectors.
         * @param BV B applied to the selected Ritz vectors.
         * @param S Selected residual block.
         * @param evals Selected Ritz values.
         * @return Correction block.
         */
        [[nodiscard]] MatrixType full_Olsen_correction(const MatrixType &V, const MatrixType &BV, const MatrixType &S, const VectorReal &evals);

        /*!
         * Compute an l2 Jacobi-Davidson correction block.
         * @param V Current Ritz vectors.
         * @param S Current residual block.
         * @param evals Current Ritz values.
         * @return Correction block.
         */
        [[nodiscard]] MatrixType jacobi_davidson_l2_correction(const MatrixType &V, const MatrixType &S, const VectorReal &evals);
        /*!
         * Compute a B-metric Jacobi-Davidson correction block.
         * @param V Current Ritz vectors.
         * @param BV Products B V.
         * @param S Current residual block.
         * @param evals Current Ritz values.
         * @return Correction block.
         */
        [[nodiscard]] MatrixType jacobi_davidson_bm_correction(const MatrixType &V, const MatrixType &BV, const MatrixType &S, const VectorReal &evals)
            requires(form_ == grit::Form::GENERALIZED);
        /*!
         * Build the correction block from the residual block.
         * @param S_in Current residual block.
         * @return Correction block.
         */
        [[nodiscard]] MatrixType get_sBlock(const MatrixType &S_in);
        /*!
         * Build the correction block for selected Ritz pairs.
         * @param V Selected Ritz vectors.
         * @param BV B applied to the selected Ritz vectors.
         * @param S Selected residual block.
         * @param evals Selected Ritz values.
         * @param V_proj Ritz vectors used by the correction projector.
         * @param BV_proj B applied to the Ritz vectors used by the correction projector.
         * @return Correction block.
         */
        [[nodiscard]] MatrixType get_sBlock(const MatrixType &V, const MatrixType &BV, const MatrixType &S, const VectorReal &evals,
                                            const MatrixType &V_proj, const MatrixType &BV_proj);

        public:
        /*! Expand or restart the GD+K search space. */
        void build() final;
        /*!
         * Append a standard-problem correction block to the basis.
         * @param Q Search basis.
         * @param AQ Products A Q.
         * @param Q_new New basis block.
         * @param AQ_new Products A Q_new.
         */
        void build(MatrixType &Q, MatrixType &AQ, const MatrixType &Q_new, const MatrixType &AQ_new);
        /*!
         * Append a generalized-problem correction block to the basis.
         * @param Q Search basis.
         * @param AQ Products A Q.
         * @param BQ Products B Q.
         * @param Q_new New basis block.
         * @param AQ_new Products A Q_new.
         * @param BQ_new Products B Q_new.
         */
        void build(MatrixType &Q, MatrixType &AQ, MatrixType &BQ, const MatrixType &Q_new, const MatrixType &AQ_new, const MatrixType &BQ_new);
    };
}

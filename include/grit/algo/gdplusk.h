#pragma once

#include <Eigen/Core>
#include <functional>
#include "grit/form/base.h"
#include <limits>
#include <optional>
#include <spdlog/common.h>
#include <type_traits>

namespace grit::algo {
    /*! Generalized Davidson plus Krylov correction solver. */
    template<typename Scalar_, grit::Form form_>
    class gdplusk : public form::base<Scalar_, form_> {
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
        using Base::qBlocks;
        using Base::rNormsRel;
        using Base::rNormScales;
        using Base::residual_correction_type_internal;
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
        using Base::cfg;
        using Base::eps;
        using Base::get_bm_normalizer_for_the_projected_pencil;
        using Base::get_op_norm_estimate;
        using Base::get_optimal_rayleigh_ritz_matrix;
        using Base::get_refined_ritz_eigenvectors_gen;
        using Base::get_refined_ritz_eigenvectors_std;
        using Base::get_residuals;
        using Base::get_standard_deviations;
        using Base::log;
        using Base::normTol;

        /*! Configuration for GD+K. */
        struct Config : BaseConfig {
            static constexpr RealScalar eps                              = std::numeric_limits<RealScalar>::epsilon(); /*!< Machine epsilon. */
            ResidualCorrectionType      residual_correction_type         = ResidualCorrectionType::NONE; /*!< Correction used to expand the search space. */
            bool                        use_adaptive_inner_tolerance     = false; /*!< Tighten the inner correction tolerance as residuals improve. */
            bool                        use_jd_b_only                    = false; /*!< Use only B in the generalized Jacobi-Davidson projector. */
            bool                        use_krylov_schur_gdplusk_restart = false; /*!< Use Krylov-Schur style restart for GD+K. */
            bool                        inject_randomness                = false; /*!< Randomize dependent correction vectors. */
            Eigen::Index                maxRetainBlocks                  = 1;     /*!< Number of Ritz blocks kept during restart compression. */
            Eigen::Index                maxPrevBlocks                    = 1;     /*!< Number of previous active Ritz blocks kept between outer iterations. */
            RealScalar                  inner_tol                        = RealScalar{0.1f};   /*!< Initial tolerance for inner correction solves. */
            Eigen::Index                inner_max_iters                  = 1000;               /*!< Maximum inner iterations in each inner correction solve. */
            RealScalar                  auto_ritz_tolerance              = RealScalar{1e-3f};  /*!< AUTO tolerance for residual-relative Ritz localization and probe progress. */
            Eigen::Index                auto_cheap_probe_interval        = 5;                  /*!< Jacobi-Davidson outer iterations between cheap Olsen probes. */
            std::function<void(const gdplusk<Scalar, form_> &)> user_callback;                 /*!< Callback called after each outer iteration. */
            Eigen::Index                                        max_extra_ritz_history    = 1; /*!< Extra Ritz history retained for progress checks. */
            Eigen::Index                                        max_ritz_residual_history = 1; /*!< Ritz residual history retained for progress checks. */
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
        Eigen::Index max_mBlocks       = 1;
        Eigen::Index max_sBlocks       = 1;
        RealScalar   current_inner_tol = RealScalar{0.1f};
        Eigen::Index vBlocks           = 0;
        Eigen::Index mBlocks           = 0;
        Eigen::Index sBlocks           = 0;
        Eigen::Index kBlocks           = 0;
        MatrixType   Q_new, AQ_new, BQ_new;
        MatrixType   G;
        void         shift_blocks_right(Eigen::Ref<MatrixType> matrix, Eigen::Index offset_old, Eigen::Index offset_new, Eigen::Index extent);
        void         roll_blocks_left(Eigen::Ref<MatrixType> matrix, Eigen::Index offset, Eigen::Index extent);
        void         selective_orthonormalize(const Eigen::Ref<const MatrixType> X, Eigen::Ref<MatrixType> Y, RealScalar breakdownTol, VectorIdxT &mask);
        void         make_new_Q_block();
        void         assert_config() const;
        void         assert_operator_config() const requires(form_ == grit::Form::STANDARD);
        void         assert_operator_config() const requires(form_ == grit::Form::GENERALIZED);
        static std::string_view       ResidualCorrectionToString(ResidualCorrectionType rct);
        static ResidualCorrectionType StringToResidualCorrection(std::string_view rct);
        void                          preamble() final;
        void                          updateStatus() final;
        void                          extractRitzVectors() final;
        void                          run_user_callback() final;

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
         * Compute the full Olsen correction block.
         * @param V Current Ritz vectors.
         * @param S Current residual block.
         * @return Correction block.
         */
        [[nodiscard]] MatrixType full_Olsen_correction(const MatrixType &V, const MatrixType &S);
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

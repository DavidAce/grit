#pragma once

#include "../enums.h"
#include "../internal/log.h"
#include "../internal/precondition/IterativeLinearSolverConfig.h"
#include "../internal/tid.h"
#include "../Matvec.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <deque>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <source_location>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace grit {
    template<typename Scalar>
    class Result;

    template<typename Scalar>
    class ResultView;
}

namespace grit::form {
    /*! Shared state and operations for standard and generalized eigensolver forms. */
    template<typename Scalar_, grit::Form form_ = grit::Form::STANDARD>
    class base {
        public:
        using Scalar     = Scalar_;                                                   /*!< Scalar type of vectors and operators. */
        using RealScalar = decltype(std::real(std::declval<Scalar>()));               /*!< Real scalar type used for Ritz values and norms. */
        using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;     /*!< Dense block of vectors. */
        using MatrixReal = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>; /*!< Dense real matrix. */
        using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;                  /*!< Dense single vector. */
        using VectorReal = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;              /*!< Real-valued vector. */
        using VectorIdxT = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;            /*!< Vector of column indices. */
        using fMultP_t   = std::function<MatrixType(const Eigen::Ref<const MatrixType> &, const Eigen::Ref<const VectorReal> &,
                                                    std::optional<const Eigen::Ref<const MatrixType>>)>; /*!< Preconditioner callback type. */

        using ResidualCorrectionType = grit::ResidualCorrectionType; /*!< Residual correction selector. */
        /*! How to handle linearly dependent candidate vectors. */
        enum class MaskPolicy {
            COMPRESS, /*!< Drop dependent columns. */
            RANDOMIZE /*!< Replace dependent columns by random vectors. */
        };
        /*! Whether operator products should be refreshed after orthogonalization. */
        enum class RefreshMult {
            YES, /*!< Refresh operator products. */
            NO   /*!< Keep existing operator products. */
        };

        static constexpr auto eps  = std::numeric_limits<RealScalar>::epsilon(); /*!< Machine epsilon. */
        static constexpr auto half = RealScalar{1} / RealScalar{2};              /*!< One half in the real scalar type. */
        static constexpr auto form = form_;                                      /*!< Problem form of this base instance. */

        /*! Configuration shared by all eigensolvers. */
        struct BaseConfig {
            Eigen::Index              nev                                     = 1;     /*!< Number of requested eigenpairs. */
            Eigen::Index              ncv                                     = 8;     /*!< Maximum search-space columns. */
            Eigen::Index              block_size                              = 2;     /*!< Number of vectors in each solver block. */
            bool                      use_b_inner_product                     = false; /*!< Use the B-metric inner product in generalized problems. */
            bool                      use_refined_rayleigh_ritz               = false; /*!< Use refined Rayleigh-Ritz extraction. */
            bool                      use_rayleigh_quotients_instead_of_evals = false; /*!< Report full Rayleigh quotients instead of projected Ritz values. */
            bool                      use_rescaled_rnorm_tolerance            = false; /*!< Rescale abstol by the current operator norm estimate. */
            Ritz                      ritz                                    = Ritz::SR;    /*!< Which Ritz values to target. */
            RealScalar                abstol                                  = eps * 10000; /*!< Absolute residual tolerance floor. */
            RealScalar                reltol                       = 0; /*!< Residual reduction from the stabilized Ritz reference; zero disables it. */
            RealScalar                ritz_stabilization_tolerance = RealScalar{1e-3f}; /*!< Tolerance for residual-scaled Ritz-value tests. */
            Eigen::Index              max_iters                    = 100l;              /*!< Maximum outer iterations; negative means unlimited. */
            Eigen::Index              max_matvecs                  = -1l;               /*!< Maximum total matrix-vector products; negative means unlimited. */
            bool                      quit_when_saturated          = true;               /*!< Stop after Ritz values and residuals remain saturated. */
            spdlog::level::level_enum log_level                    = spdlog::level::warn; /*!< Solver log level. */
        };

        /*! Orthogonalization diagnostics for l2 and B-metric blocks. */
        struct OrthMeta {
            MatrixType Gram;                                                         /*!< Gram matrix used in the last orthogonalization check. */
            MatrixType Gram_symm;                                                    /*!< Hermitian part of the Gram matrix. */
            MatrixType Gram_skew;                                                    /*!< Skew-Hermitian part of the Gram matrix. */
            MatrixType Gram_skew_fwd;                                                /*!< Forward skew diagnostic. */
            MatrixType Gram_skew_adj;                                                /*!< Adjoint skew diagnostic. */
            VectorReal Rdiag;                                                        /*!< Diagonal of the triangular factor from orthonormalization. */
            RealScalar maskTol       = std::numeric_limits<RealScalar>::quiet_NaN(); /*!< Tolerance for masking dependent columns. */
            RealScalar orthTol       = std::numeric_limits<RealScalar>::quiet_NaN(); /*!< Orthogonality tolerance used in the last check. */
            RealScalar skewTol       = std::pow(eps, RealScalar{0.25f});             /*!< Skew tolerance used for Gram diagnostics. */
            RealScalar orthError     = std::numeric_limits<RealScalar>::quiet_NaN(); /*!< Measured orthogonality error. */
            RealScalar symmError     = std::numeric_limits<RealScalar>::quiet_NaN(); /*!< Measured Hermitian symmetry error. */
            RealScalar skewError     = std::numeric_limits<RealScalar>::quiet_NaN(); /*!< Measured skew error. */
            RealScalar skewError_fwd = std::numeric_limits<RealScalar>::quiet_NaN(); /*!< Measured forward skew error. */
            RealScalar skewError_adj = std::numeric_limits<RealScalar>::quiet_NaN(); /*!< Measured adjoint skew error. */
            VectorReal proj_sum_a;                                                   /*!< Projection coefficients from the first orthogonalization pass. */
            VectorReal proj_sum_b;                                                   /*!< Projection coefficients from the second orthogonalization pass. */
            VectorReal scale_log;                                                    /*!< Column scaling diagnostics in logarithmic form. */
            VectorIdxT mask;                                                         /*!< Kept-column mask after orthonormalization. */
            bool       force_refresh_a           = false;                            /*!< Refresh A times the block after orthogonalization. */
            bool       refresh_by                = true;                             /*!< Refresh B times the block after orthogonalization. */
            bool       gram_is_positive_definite = false;                            /*!< Whether the last Gram matrix passed the positive-definite check. */
            MaskPolicy maskPolicy                = MaskPolicy::RANDOMIZE;            /*!< Policy for dependent columns. */

            /*!
             * Analyze l2 orthonormality of Y.
             * @param Y Block to check.
             */
            void analyze_l2_orthonormality(const Eigen::Ref<const MatrixType> &Y);
            /*!
             * Analyze B-orthonormality using Y and B Y.
             * @param Y Block to check.
             * @param BY Product B Y.
             */
            void analyze_b_orthonormality(const Eigen::Ref<const MatrixType> &Y, const Eigen::Ref<const MatrixType> &BY);
            /*!
             * Analyze B-metric orthonormality using Y and B Y.
             * @param Y Block to check.
             * @param BY Product B Y.
             */
            void analyze_bm_orthonormality(const Eigen::Ref<const MatrixType> &Y, const Eigen::Ref<const MatrixType> &BY);
            /*!
             * Analyze l2 orthogonality between X and Y.
             * @param X First block.
             * @param Y Second block.
             */
            void analyze_l2_orthogonality(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &Y);
            /*!
             * Analyze B-metric orthogonality between X and Y.
             * @param X First block.
             * @param BX Product B X.
             * @param Y Second block.
             * @param B_Y Product B Y.
             */
            void analyze_bm_orthogonality(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &BX, const Eigen::Ref<const MatrixType> &Y,
                                          const Eigen::Ref<const MatrixType> &B_Y);
        };

        /*! Solver output, counters, timers, and stopping diagnostics. */
        struct Status {
            VectorReal                eigVal;                                                          /*!< Current selected Ritz values. */
            VectorReal                oldVal;                                                          /*!< Ritz values from the previous status update. */
            VectorReal                absDiff;                                                         /*!< Absolute Ritz-value changes. */
            VectorReal                relDiff;                                                         /*!< Relative Ritz-value changes. */
            RealScalar                initVal          = std::numeric_limits<RealScalar>::quiet_NaN(); /*!< Initial Ritz value used for progress checks. */
            RealScalar                condition_a = RealScalar{1}; /*!< Euclidean condition estimate of A on the search subspace. */
            RealScalar                condition_b = RealScalar{1}; /*!< Euclidean condition estimate of B on the search subspace. */
            RealScalar                op_norm_estimate = RealScalar{1};                                /*!< Current operator norm estimate. */
            RealScalar                gap              = std::numeric_limits<RealScalar>::infinity();  /*!< Current Ritz gap estimate. */
            std::vector<Eigen::Index> optIdx;                                   /*!< Indices of selected Ritz values in the projected problem. */
            Eigen::Index              outer_iter                           = 0; /*!< Outer iteration count. */
            Eigen::Index              num_outer_iters_last_restart         = 0; /*!< Outer iteration count at the last restart. */
            Eigen::Index              num_inner_iters                      = 0; /*!< Inner correction iterations in the last outer iteration. */
            Eigen::Index              num_inner_iters_prev                 = 0; /*!< Inner correction iterations in the previous outer iteration. */
            Eigen::Index              num_inner_iters_total                = 0; /*!< Total inner correction iterations. */
            Eigen::Index              num_matvecs_inner                    = 0; /*!< A and B matrix-vector products in the last inner solve. */
            Eigen::Index              num_matvecs_inner_total              = 0; /*!< Total inner A and B matrix-vector products. */
            Eigen::Index              num_matvecs_a                        = 0; /*!< A matrix-vector products in the last outer iteration. */
            Eigen::Index              num_matvecs_b                        = 0; /*!< B matrix-vector products in the last outer iteration. */
            Eigen::Index              num_matvecs_a_inner                  = 0; /*!< A matrix-vector products in the last inner solve. */
            Eigen::Index              num_matvecs_b_inner                  = 0; /*!< B matrix-vector products in the last inner solve. */
            Eigen::Index              num_matvecs_a_total                  = 0; /*!< Total A matrix-vector products. */
            Eigen::Index              num_matvecs_b_total                  = 0; /*!< Total B matrix-vector products. */
            Eigen::Index              num_operator_inner                   = 0; /*!< Projected correction-operator applications in the last inner solve. */
            Eigen::Index              num_operator_inner_total             = 0; /*!< Total projected correction-operator applications. */
            Eigen::Index              num_matvecs                          = 0; /*!< Matrix-vector products in the last outer iteration. */
            Eigen::Index              num_matvecs_total                    = 0; /*!< Total matrix-vector products. */
            Eigen::Index              num_precond                          = 0; /*!< Preconditioner applications in the last outer iteration. */
            Eigen::Index              num_precond_inner                    = 0; /*!< Projected preconditioner applications in the last inner solve. */
            Eigen::Index              num_precond_inner_total              = 0; /*!< Total projected inner preconditioner applications. */
            Eigen::Index              num_precond_total                    = 0; /*!< Total preconditioner applications. */
            Eigen::Index              num_preconditioner_updates           = 0; /*!< Preconditioner updates in the last outer iteration. */
            Eigen::Index              num_preconditioner_updates_inner     = 0; /*!< Preconditioner updates for the last inner correction. */
            Eigen::Index              num_preconditioner_updates_total     = 0; /*!< Total preconditioner updates. */
            Eigen::Index              num_preconditioner_apply_inner       = 0; /*!< User preconditioner callback applications in the last inner solve. */
            Eigen::Index              num_preconditioner_apply_inner_total = 0; /*!< Total user preconditioner callback applications in inner solves. */
            Eigen::Index              num_preconditioner_apply_total       = 0; /*!< Total user preconditioner callback applications. */
            tid::ur                   time_elapsed;                             /*!< Total solver wall timer. */
            tid::ur                   time_solve_inner;                         /*!< Complete inner linear-solve wall time. */
            tid::ur                   time_matvecs;                             /*!< Outer A and B matrix-vector product time. */
            tid::ur                   time_matvecs_a;                           /*!< Outer A matrix-vector callback time. */
            tid::ur                   time_matvecs_b;                           /*!< Outer B matrix-vector callback time. */
            tid::ur                   time_matvecs_a_inner;                     /*!< Inner A matrix-vector callback time. */
            tid::ur                   time_matvecs_b_inner;                     /*!< Inner B matrix-vector callback time. */
            tid::ur                   time_precond;                             /*!< Outer preconditioner callback time. */
            tid::ur                   time_preconditioner_inner;                /*!< Inner projected-preconditioner time, including projectors. */
            tid::ur                   time_preconditioner_update;               /*!< Outer preconditioner-update callback time. */
            tid::ur                   time_preconditioner_update_inner;         /*!< Inner preconditioner-update callback time. */
            tid::ur                   time_preconditioner_apply_inner;          /*!< Inner user preconditioner callback time, excluding projectors. */
            tid::ur                   time_operator_inner;                /*!< Projected correction-operator time, including projectors and A/B callbacks. */
            tid::ur                   time_project_left_inner;            /*!< Inner left-projector time. */
            tid::ur                   time_project_right_inner;           /*!< Inner right-projector time. */
            tid::ur                   time_residual_correction;           /*!< Residual-correction construction time. */
            tid::ur                   time_build;                         /*!< Search-space build time, including correction and orthogonalization. */
            tid::ur                   time_orthogonalize;                 /*!< Timer for orthogonalization. */
            tid::ur                   time_orthonormalize;                /*!< Timer for orthonormalization. */
            tid::ur                   time_orth_project;                  /*!< Timer for orthogonalization projections. */
            tid::ur                   time_orth_factor;                   /*!< Timer for orthonormalization factorizations. */
            tid::ur                   time_orth_update;                   /*!< Timer for orthogonalization updates. */
            tid::ur                   time_orth_refresh;                  /*!< Timer for refreshing operator products. */
            tid::ur                   time_orth_mask;                     /*!< Timer for masking dependent vectors. */
            tid::ur                   time_diagonalize;                   /*!< Timer for projected-problem diagonalization. */
            tid::ur                   time_extract_ritz;                  /*!< Timer for Ritz extraction. */
            tid::ur                   time_restart;                       /*!< Timer for search-space restarts. */
            tid::ur                   time_status_update;                 /*!< Timer for convergence and status updates. */
            RealScalar                inner_error_last   = RealScalar{0}; /*!< Last inner correction residual. */
            RealScalar                inner_tol_last     = RealScalar{0}; /*!< Last inner correction tolerance. */
            bool                      residual_converged = false;         /*!< Whether selected residuals satisfy the active tolerance. */
            bool                      residual_below_gap = false;         /*!< Whether selected residuals are below the Ritz gap criterion. */
            VectorReal                rNormsAbs;                          /*!< Current selected absolute residual norms. */
            VectorReal                rnorm_abs_reference;                /*!< Residual norms recorded when the selected Ritz values stabilize. */
            std::deque<VectorReal>    rNormsAbsHistory;                   /*!< Recent absolute residual-norm history. */
            std::deque<VectorReal>    eigVals_history;                    /*!< Recent Ritz-value history. */
            std::deque<Eigen::Index>  matvecs_history;                    /*!< Recent matrix-vector count history. */
            size_t                    max_history_size        = 12;       /*!< Maximum stored history length. */
            Eigen::Index              saturation_count_eigVal = 0;        /*!< Consecutive eigenvalue saturation count. */
            Eigen::Index              saturation_count_rNorm  = 0;        /*!< Consecutive residual saturation count. */
            Eigen::Index              saturation_count_max    = 10;       /*!< Saturation count required before stopping. */
            std::vector<std::string>  stopMessage             = {};       /*!< Human-readable stop messages. */
            StopReason                stopReason              = StopReason::none; /*!< Solver stop reason. */
            Ritz                      ritz_internal           = Ritz::NONE;       /*!< Effective Ritz selector used internally. */
        };

        /*! State used by AUTO residual correction. */
        struct AutoResidualCorrectionState {
            ResidualCorrectionType active            = ResidualCorrectionType::CHEAP_OLSEN; /*!< Correction currently preferred by AUTO. */
            ResidualCorrectionType iteration_method  = ResidualCorrectionType::CHEAP_OLSEN; /*!< Correction used in the current outer iteration. */
            Eigen::Index           cheap_olsen_iters = 0;                                   /*!< Consecutive cheap Olsen outer iterations. */
            Eigen::Index jd_outer_iters_since_probe  = 0;   /*!< Number of outer iterations using JD corrections since the last cheap Olsen correction. */
            Eigen::Index probes_started              = 0;   /*!< AUTO probes started in the current stabilized Ritz basin. */
            double       outer_iteration_time_start  = 0.0; /*!< Wall-time marker for the current outer iteration. */
            std::vector<Eigen::Index> cheap_olsen_to_jd_switch_outer_iters; /*!< Outer iterations switching from cheap Olsen to Jacobi-Davidson. */
            std::vector<Eigen::Index> jd_to_cheap_olsen_switch_outer_iters; /*!< Outer iterations switching from Jacobi-Davidson to cheap Olsen. */
        };

        /*! Convert a residual correction selector to text. */
        static std::string_view ResidualCorrectionToString(ResidualCorrectionType rct);
        /*! Convert text to a residual correction selector. */
        static ResidualCorrectionType StringToResidualCorrection(std::string_view rct);

        protected:
        BaseConfig           default_cfg = {};
        BaseConfig          *cfg_ptr     = &default_cfg;
        Logger::LoggerHandle log;

        Eigen::Index qBlocks = 0;

        void                            bind_config(BaseConfig &cfg);
        [[nodiscard]] BaseConfig       &cfg();
        [[nodiscard]] const BaseConfig &cfg() const;
        [[nodiscard]] const MatrixType &initial_guess() const;

        public:
        virtual ~base() = default;

        /*!
         * Set the solver logger level and optional logger name.
         * @param logLevel Log level used by this solver.
         * @param name Optional logger name suffix.
         */
        void setLogger(spdlog::level::level_enum logLevel, const std::string &name = "");
        /*!
         * Set the initial search vectors.
         * @param V Initial vectors stored in columns.
         */
        void set_initial_guess(MatrixType V);
        /*! Remove the stored initial guess. */
        void clear_initial_guess();
        /*! Whether an initial guess has been stored. */
        [[nodiscard]] bool has_initial_guess() const;
        /*! Return a non-owning read-only view of the current solver result. */
        [[nodiscard]] grit::ResultView<Scalar> get_result_view() const;
        /*! Return an owning snapshot of the current solver result. */
        [[nodiscard]] grit::Result<Scalar> get_result() const;

        /*!
         * Construct a standard form.
         * @param V Initial vectors stored in columns.
         * @param A Matrix-free operator A.
         */
        base(const MatrixType &V, Matvec<Scalar> &A) requires(form_ == grit::Form::STANDARD);
        /*!
         * Construct a generalized form.
         * @param V Initial vectors stored in columns.
         * @param A Matrix-free operator A.
         * @param B Matrix-free operator B.
         */
        base(const MatrixType &V, Matvec<Scalar> &A, Matvec<Scalar> &B) requires(form_ == grit::Form::GENERALIZED);

        Status                 status = {};                               /*!< Current solver status and diagnostics. */
        Eigen::Index           N;                                         /*!< Operator dimension. */
        Eigen::Index           size;                                      /*!< Alias for the operator dimension. */
        bool                   dev_append_extra_blocks_to_basis  = false; /*!< Development option to retain extra candidate blocks. */
        bool                   dev_skipjcb                       = false; /*!< Development option to skip Jacobi-Davidson correction blocks. */
        int                    chebyshev_filter_degree           = 0;     /*!< Degree of the optional Chebyshev filter. */
        ResidualCorrectionType residual_correction_type_internal = ResidualCorrectionType::NONE; /*!< Effective residual correction used internally. */
        Matvec<Scalar>        &A;                                                                /*!< Matrix-free operator A. */
        std::optional<std::reference_wrapper<Matvec<Scalar>>> B = std::nullopt;   /*!< Optional matrix-free operator B for generalized problems. */
        MatrixType                                            T;                  /*!< Projected matrix for standard problems. */
        MatrixType                                            Aproj, Bproj, W, Q; /*!< Projected matrices, residual block, and search basis. */
        MatrixType                                            AQ, BQ;             /*!< Operator products A Q and B Q. */
        MatrixType                                            V;                  /*!< Selected Ritz vectors. */
        MatrixType                                            AV;                 /*!< Products A V for selected Ritz vectors. */
        MatrixType                                            BV;                 /*!< Products B V for selected Ritz vectors. */
        MatrixType                                            V_prev;             /*!< Selected Ritz vectors from the previous outer iteration. */
        MatrixType                                            K, K_prev;          /*!< Krylov or correction blocks. */
        MatrixType                                            S, S1, S2;          /*!< Residual and correction blocks. */
        MatrixType                                            D;                  /*!< Search directions or diagonal block. */
        MatrixType                                            M, AM, BM;          /*!< Auxiliary basis block and operator products. */
        VectorReal                                            T_evals;            /*!< Ritz values of the projected problem. */
        MatrixType                                            T1, T2, T_evecs;    /*!< Generalized projected pencil and Ritz vectors. */
        Eigen::HouseholderQR<MatrixType>                      hhqr;               /*!< Householder QR workspace. */

        RealScalar                  skewTol         = std::sqrt(eps) * 10000;                     /*!< Tolerance for skew-Hermitian Gram diagnostics. */
        RealScalar                  normTol         = eps * 10;                                   /*!< Tolerance for detecting small vector norms. */
        RealScalar                  orthTol         = eps * 100;                                  /*!< Tolerance for orthogonality checks. */
        RealScalar                  quotTolB        = RealScalar{1e-10f};                         /*!< Tolerance for small B-Rayleigh quotient denominators. */
        RealScalar                  rnormRelDiffTol = std::numeric_limits<RealScalar>::epsilon(); /*!< Relative residual-change tolerance. */
        RealScalar                  absDiffTol      = std::numeric_limits<RealScalar>::epsilon() * 10000; /*!< Absolute Ritz-value change tolerance. */
        RealScalar                  relDiffTol      = std::numeric_limits<RealScalar>::epsilon() * 10000; /*!< Relative Ritz-value change tolerance. */
        std::string                 tag;                                                                  /*!< Short solver tag used in log messages. */
        AutoResidualCorrectionState auto_residual_correction;                                             /*!< AUTO residual correction state. */

        /*! Return the problem form name. */
        std::string_view form_name() const;

        /*!
         * Compute residuals for candidate Ritz pairs.
         * @param Y Ritz values.
         * @param AV Products A V.
         * @param BV Products B V, or an empty block for standard problems.
         * @param rNormsAbs Output residual norms.
         * @return Residual block.
         */
        MatrixType get_residuals(const Eigen::Ref<const VectorReal> &Y, const Eigen::Ref<const MatrixType> &AV, const Eigen::Ref<const MatrixType> &BV,
                                 VectorReal &rNormsAbs);
        /*!
         * Absolute residual target for the nth selected eigenpair.
         * @param n Selected eigenpair index.
         * @return Computed convergence target max(reltol * stabilized reference, abstol * scale).
         */
        RealScalar rNormAbsTarget(Eigen::Index n) const;
        /*! Absolute residual targets for all selected eigenpairs. */
        VectorReal rNormAbsTargets() const;
        /*!
         * Residual scale for the nth selected eigenpair.
         * @param n Selected eigenpair index.
         * @return Scale used to turn an absolute residual into a rescaled residual.
         */
        RealScalar rNormScale(Eigen::Index n) const;
        /*! Residual scales for all selected eigenpairs. */
        VectorReal rNormScales() const;
        /*!
         * Convert absolute residual norms to rescaled residual norms.
         * @param rNormsAbs Absolute residual norms.
         * @return Rescaled residual norms.
         */
        VectorReal rNormsRel(const VectorReal &rNormsAbs) const;
        /*! Log10 residual-norm change per matrix-vector product. */
        RealScalar get_rNorms_log10_change_per_matvec();
        /*!
         * Estimate the operator norm, optionally using a Ritz value.
         * @param eigval Ritz value used as a lower bound when supplied.
         * @return Operator norm estimate.
         */
        RealScalar get_op_norm_estimate(std::optional<RealScalar> eigval = std::nullopt) const;

        /*!
         * Apply A to a block of vectors and update counters.
         * @param X Input block.
         * @return Product A X.
         */
        MatrixType MultA(const Eigen::Ref<const MatrixType> &X);
        /*!
         * Apply A inside an inner correction solve.
         * @param X Input block.
         * @return Product A X.
         */
        MatrixType MultA_inner(const Eigen::Ref<const MatrixType> &X);
        /*!
         * Apply the preconditioner to a block of residual vectors.
         * @param X Residual or correction block.
         * @param evals Current Ritz values.
         * @return Preconditioned block.
         */
        MatrixType MultP(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const VectorReal> &evals);

        /*!
         * Apply B to a block of vectors and update counters.
         * @param X Input block.
         * @return Product B X.
         */
        MatrixType MultB(const Eigen::Ref<const MatrixType> &X) requires(form_ == grit::Form::GENERALIZED);
        /*!
         * Apply B inside an inner correction solve.
         * @param X Input block.
         * @return Product B X.
         */
        MatrixType MultB_inner(const Eigen::Ref<const MatrixType> &X) requires(form_ == grit::Form::GENERALIZED);

        /*!
         * Select Ritz indices from the projected spectrum.
         * @param ritz Ritz selector.
         * @param offset Number of selected values to skip.
         * @param num Number of indices to return.
         * @param evals Projected Ritz values.
         * @return Selected projected indices.
         */
        std::vector<Eigen::Index> get_ritz_indices(Ritz ritz, Eigen::Index offset, Eigen::Index num, const VectorReal &evals) const;

        /*! Initialize solver storage before a run. */
        void init();
        /*! Expand or restart the search space. */
        virtual void build() = 0;
        /*! Diagonalize the current projected problem. */
        void diagonalizeT();

        /*!
         * Return indices of selected entries after partial sorting.
         * @param v Values to sort.
         * @param offset Number of sorted entries to skip.
         * @param num Number of indices to return.
         * @param comp Ordering predicate.
         * @return Selected indices.
         */
        template<typename Comp>
        std::vector<Eigen::Index> getIndices(const VectorReal &v, const Eigen::Index offset, const Eigen::Index num, Comp comp) const {
            std::vector<Eigen::Index> idx(static_cast<size_t>(v.size()));
            Eigen::Index              numSort = std::min<Eigen::Index>(offset + num, v.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + numSort, idx.end(), [&](auto i, auto j) { return comp(v(i), v(j)); });
            return std::vector(idx.begin() + offset, idx.begin() + numSort);
        }

        /*! Extract selected Ritz vectors and residuals from the projected problem. */
        virtual void extractRitzVectors();
        /*!
         * Extract Ritz vectors for a standard problem.
         * @param optIdx Selected projected indices.
         * @param V Output Ritz vectors.
         * @param AV Output products A V.
         * @param S Output residual block.
         * @param rNormsAbs Output residual norms.
         */
        void extractRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &AV, MatrixType &S, VectorReal &rNormsAbs);
        /*!
         * Extract Ritz vectors for a generalized problem.
         * @param optIdx Selected projected indices.
         * @param V Output Ritz vectors.
         * @param AV Output products A V.
         * @param BV Output products B V.
         * @param S Output residual block.
         * @param rNormsAbs Output residual norms.
         */
        void extractRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &AV, MatrixType &BV, MatrixType &S, VectorReal &rNormsAbs)
            requires(form_ == grit::Form::GENERALIZED);
        /*!
         * Finish generalized Ritz extraction with an order-preserving B-metric cleanup.
         *
         * Ordinary and refined extraction already choose and order the Ritz vectors through
         * optIdx. Outside restart, GD+K relies on that order: the first columns are the active
         * Ritz block, and any extra columns are retained history. This function therefore does
         * not solve another projected problem and does not rotate the block. Its job is only to
         * make the returned vectors safe to use in B-metric orthogonalization:
         *
         *   V^* B V ~= I, with the input column order preserved.
         *
         * Small loss of B orthonormality can arise from rounding or cancellation in B V. When
         * that happens, this function uses the DGKS B orthonormalizer, which preserves column
         * order. If DGKS compresses dependent columns, the reduced column count is left for the
         * caller to handle; no replacement vectors are invented here.
         *
         * Control flow:
         * 1. Return unchanged for empty blocks or when the B inner product is disabled.
         * 2. Check that V, AV, BV, S, rNormsAbs, and optIdx describe the same Ritz block.
         * 3. If V^* B V is not close enough to I, refresh BV and run order-preserving DGKS.
         * 4. Recompute residuals, residual norms, status.optIdx, and the matching T_evals
         *    entries for the surviving columns.
         * 5. Assert the final B-orthonormality invariant.
         *
         * @param optIdx Projected Ritz indices used to form the input block.
         * @param V Ritz vectors, modified in place.
         * @param AV Products A V, modified in place.
         * @param BV Products B V, modified in place.
         * @param S Residual block, modified in place.
         * @param rNormsAbs Residual norms, modified in place.
         */
        void finalize_bm_ritz_vectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &AV, MatrixType &BV, MatrixType &S,
                                      VectorReal &rNormsAbs) requires(form_ == grit::Form::GENERALIZED);
        /*!
         * Orthonormalize projected eigenvectors in the projected metric.
         * @param Z Projected eigenvectors, modified in place.
         * @param T2 Projected metric matrix.
         */
        void orthonormalize_Z(Eigen::Ref<MatrixType> Z, const Eigen::Ref<const MatrixType> &T2);
        /*!
         * Compute refined Ritz vectors for a standard problem.
         * @param Z Projected Ritz vectors.
         * @param Y Ritz values.
         * @param Q Search basis.
         * @param AQ Products A Q.
         * @return Refined projected vectors.
         */
        MatrixType get_refined_ritz_eigenvectors_std(const Eigen::Ref<const MatrixType> &Z, const Eigen::Ref<const VectorReal> &Y, const MatrixType &Q,
                                                     const MatrixType &AQ);
        /*!
         * Compute refined Ritz vectors for a generalized problem.
         * @param Z Projected Ritz vectors.
         * @param Y Ritz values.
         * @param AQ Products A Q.
         * @param BQ Products B Q.
         * @return Refined projected vectors.
         */
        MatrixType get_refined_ritz_eigenvectors_gen(const Eigen::Ref<const MatrixType> &Z, const Eigen::Ref<const VectorReal> &Y, const MatrixType &AQ,
                                                     const MatrixType &BQ) requires(form_ == grit::Form::GENERALIZED);
        /*!
         * Compute a B-metric normalizer for the projected generalized pencil.
         * @param T2 Projected B matrix.
         * @return Normalizer and normalized metric matrix.
         */
        std::pair<MatrixType, MatrixType> get_bm_normalizer_for_the_projected_pencil(const MatrixType &T2);
        /*!
         * Pick the better projected vector between Rayleigh-Ritz and refined candidates.
         * @param Z_rr Rayleigh-Ritz projected vectors.
         * @param Z_ref Refined projected vectors.
         * @param T1 Projected A matrix.
         * @param T2 Projected B matrix.
         * @return Projected vectors with the smaller residuals.
         */
        MatrixType get_optimal_rayleigh_ritz_matrix(const MatrixType &Z_rr, const MatrixType &Z_ref, const MatrixType &T1, const MatrixType &T2);
        /*!
         * Apply refined Rayleigh-Ritz extraction for a generalized problem.
         * @param optIdx Selected projected indices.
         * @param V Output Ritz vectors.
         * @param AV Output products A V.
         * @param BV Output products B V.
         * @param S Output residual block.
         * @param rNormsAbs Output residual norms.
         */
        void refinedRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &AV, MatrixType &BV, MatrixType &S, VectorReal &rNormsAbs)
            requires(form_ == grit::Form::GENERALIZED);
        /*!
         * Apply refined Rayleigh-Ritz extraction for a standard problem.
         * @param optIdx Selected projected indices.
         * @param V Output Ritz vectors.
         * @param AV Output products A V.
         * @param S Output residual block.
         * @param rNormsAbs Output residual norms.
         */
        void refinedRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &AV, MatrixType &S, VectorReal &rNormsAbs);
        /*! Apply refined Rayleigh-Ritz extraction to the current selected vectors. */
        void refinedRitzVectors();
        /*! Refresh selected Ritz products and residuals from direct operator applications. */
        void refresh_direct_ritz_residuals();
        /*! Return restart diagnostics from direct operator applications. */
        std::string get_direct_ritz_diagnostics();
        /*! Update Euclidean condition estimates for A and B on the current search space. */
        void update_condition_numbers();
        /*! Prepare a solver run. */
        virtual void preamble();
        /*! Update status after an outer iteration. */
        virtual void updateStatus();
        /*! Print the current solver status. */
        void printStatus();
        /*! Print a final one-line summary per requested eigenpair. */
        void printFinal();
        /*! Finish accumulated timer laps for the current outer iteration. */
        void restart_status_time_laps();
        /*! Call the user callback, if present. */
        virtual void run_user_callback();

        /*! Perform one outer iteration. */
        void do_outer_iteration();
        /*! Run the solver until convergence or a stop condition is reached. */
        void run();

        /*!
         * Assert that all entries are finite.
         * @param X Matrix to check.
         * @param location Source location used in the error message.
         */
        void assert_allFinite(const Eigen::Ref<const MatrixType> &X, const std::source_location &location = std::source_location::current());
        /*!
         * Assert l2 orthonormality using stored diagnostics.
         * @param X Matrix checked by the diagnostics.
         * @param m Orthogonalization diagnostics.
         * @param location Source location used in the error message.
         */
        void assert_l2_orthonormal(const Eigen::Ref<const MatrixType> &X, const OrthMeta &m,
                                   const std::source_location &location = std::source_location::current());
        /*!
         * Assert l2 orthogonality using stored diagnostics.
         * @param X First block.
         * @param Y Second block.
         * @param m Orthogonalization diagnostics.
         * @param location Source location used in the error message.
         */
        void assert_l2_orthogonal(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &Y, const OrthMeta &m,
                                  const std::source_location &location = std::source_location::current());
        /*!
         * Estimate how much the raw dot-product floor should be inflated for a B-product block.
         *
         * The returned value is an inflation factor >= 1. It compares a practical operator-scale
         * estimate against the smallest finite local B-Rayleigh scale seen in the block. The ratio
         * is a proxy for how much cancellation the B product is hiding from the dot products used
         * by the orthogonality tests, so the result is applied directly as a tolerance multiplier.
         *
         * @param Y Block used for the local B-Rayleigh scale.
         * @param BY Product B Y.
         * @return Cancellation multiplier for the B-metric tolerance floor.
         */
        RealScalar bm_cancellation_multiplier(const Eigen::Ref<const MatrixType> &Y, const Eigen::Ref<const MatrixType> &BY) const;
        /*!
         * Assert B-metric orthonormality using stored diagnostics.
         * @param X Block to check.
         * @param BX Product B X.
         * @param m Orthogonalization diagnostics.
         * @param location Source location used in the error message.
         */
        void assert_bm_orthonormal(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &BX, const OrthMeta &m,
                                   const std::source_location &location = std::source_location::current());
        /*!
         * Assert B-metric orthogonality using stored diagnostics.
         * @param X First block.
         * @param BX Product B X.
         * @param Y Second block.
         * @param BY Product B Y.
         * @param m Orthogonalization diagnostics.
         * @param location Source location used in the error message.
         */
        void assert_bm_orthogonal(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &BX, const Eigen::Ref<const MatrixType> &Y,
                                  const Eigen::Ref<const MatrixType> &BY, const OrthMeta &m,
                                  const std::source_location &location = std::source_location::current());
        /*!
         * Keep only columns selected by the mask.
         * @param X Matrix compressed in place.
         * @param mask Indices of columns to keep.
         */
        void compress_cols(MatrixType &X, const VectorIdxT &mask);
        /*!
         * Standard deviations over recent history.
         * @param v History values.
         * @param apply_log10 Apply log10 before measuring deviations.
         * @param last_n Number of trailing entries to use; negative uses the complete history.
         * @return Standard deviations per selected Ritz pair.
         */
        VectorReal get_standard_deviations(const std::deque<VectorReal> &v, bool apply_log10, Eigen::Index last_n = -1);
        /*!
         * Least-squares slopes over recent history.
         * @param v History values.
         * @param apply_log10 Apply log10 before fitting.
         * @param last_n Number of trailing entries to use; negative uses the complete history.
         * @param slope_errors Optional standard errors of the fitted slopes; NaNs when fewer than three entries are selected.
         * @return Slopes per selected Ritz pair, or NaNs when fewer than two entries are selected.
         */
        VectorReal get_slopes(const std::deque<VectorReal> &v, bool apply_log10, Eigen::Index last_n = -1, VectorReal *slope_errors = nullptr);
        /*! Whether residual norms have saturated by the configured criterion. */
        bool rNorms_have_saturated();
        /*! Whether Ritz values have saturated by the configured criterion. */
        bool eigVals_have_saturated();
        /*!
         * l2-orthogonalize Y against X and refresh A Y when requested.
         * @param X Existing basis block.
         * @param AX Products A X.
         * @param Y Candidate block modified in place.
         * @param AY Products A Y modified in place.
         * @param m Orthogonalization diagnostics.
         * @param refresh_mult Whether to refresh operator products.
         */
        void block_l2_orthogonalize(const MatrixType &X, const MatrixType &AX, MatrixType &Y, MatrixType &AY, OrthMeta &m,
                                    RefreshMult refresh_mult = RefreshMult::YES);
        /*!
         * l2-orthogonalize Y against X and refresh A Y and B Y when requested.
         * @param X Existing basis block.
         * @param AX Products A X.
         * @param BX Products B X.
         * @param Y Candidate block modified in place.
         * @param AY Products A Y modified in place.
         * @param BY Products B Y modified in place.
         * @param m Orthogonalization diagnostics.
         * @param refresh_mult Whether to refresh operator products.
         */
        void block_l2_orthogonalize(const MatrixType &X, const MatrixType &AX, const MatrixType &BX, MatrixType &Y, MatrixType &AY, MatrixType &BY, OrthMeta &m,
                                    RefreshMult refresh_mult = RefreshMult::YES);
        /*!
         * l2-orthonormalize Y and refresh A Y.
         * @param Y Candidate block modified in place.
         * @param AY Products A Y modified in place.
         * @param m Orthogonalization diagnostics.
         */
        void block_l2_orthonormalize(MatrixType &Y, MatrixType &AY, OrthMeta &m);
        /*!
         * l2-orthonormalize Y and refresh A Y and B Y.
         * @param Y Candidate block modified in place.
         * @param AY Products A Y modified in place.
         * @param BY Products B Y modified in place.
         * @param m Orthogonalization diagnostics.
         */
        void block_l2_orthonormalize(MatrixType &Y, MatrixType &AY, MatrixType &BY, OrthMeta &m);
        /*!
         * B-metric orthogonalize Y against X and refresh products when requested.
         * @param X Existing basis block.
         * @param AX Products A X.
         * @param BX Products B X.
         * @param Y Candidate block modified in place.
         * @param AY Products A Y modified in place.
         * @param BY Products B Y modified in place.
         * @param m Orthogonalization diagnostics.
         * @param refresh_mult Whether to refresh operator products.
         */
        void block_bm_orthogonalize(const MatrixType &X, const MatrixType &AX, const MatrixType &BX, MatrixType &Y, MatrixType &AY, MatrixType &BY, OrthMeta &m,
                                    RefreshMult refresh_mult = RefreshMult::YES) requires(form_ == grit::Form::GENERALIZED);
        /*!
         * B-metric orthonormalize Y and refresh A Y and B Y.
         * @param Y Candidate block modified in place.
         * @param AY Products A Y modified in place.
         * @param BY Products B Y modified in place.
         * @param m Orthogonalization diagnostics.
         */
        void block_bm_orthonormalize(MatrixType &Y, MatrixType &AY, MatrixType &BY, OrthMeta &m) requires(form_ == grit::Form::GENERALIZED);
        /*!
         * B-metric orthonormalize Y through the small Gram eigensystem and refresh A Y.
         * @param Y Candidate block modified in place.
         * @param AY Products A Y modified in place.
         * @param BY Products B Y modified in place.
         * @param m Orthogonalization diagnostics.
         */
        void block_bm_orthonormalize_eig(MatrixType &Y, MatrixType &AY, MatrixType &BY, OrthMeta &m) requires(form_ == grit::Form::GENERALIZED);
    };
}

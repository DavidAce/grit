#pragma once

#include "../enums.h"
#include "../internal/log.h"
#include "../internal/precondition/IterativeLinearSolverConfig.h"
#include "../internal/tid.h"
#include "../MatVec.h"
#include "../result.h"
#include <algorithm>
#include <complex>
#include <cmath>
#include <deque>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <fmt/format.h>
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

namespace grit::form {
    template<typename Scalar_>
    class base {
        public:
        using Scalar     = Scalar_;
        using RealScalar = decltype(std::real(std::declval<Scalar>()));
        using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using MatrixReal = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using VectorReal = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
        using VectorIdxT = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;
        using fMultP_t   = std::function<MatrixType(const Eigen::Ref<const MatrixType> &, const Eigen::Ref<const VectorReal> &,
                                                    std::optional<const Eigen::Ref<const MatrixType>>)>;

        enum class ResidualCorrectionType { NONE, CHEAP_OLSEN, FULL_OLSEN, JACOBI_DAVIDSON, AUTO };
        enum class MaskPolicy { COMPRESS, RANDOMIZE };

        static constexpr auto eps  = std::numeric_limits<RealScalar>::epsilon();
        static constexpr auto half = RealScalar{1} / RealScalar{2};

        struct OrthMeta {
            MatrixType Gram;
            MatrixType Gram_symm;
            MatrixType Gram_skew;
            MatrixType Gram_skew_fwd;
            MatrixType Gram_skew_adj;
            VectorReal Rdiag;
            RealScalar maskTol       = std::numeric_limits<RealScalar>::quiet_NaN();
            RealScalar orthTol       = std::numeric_limits<RealScalar>::quiet_NaN();
            RealScalar skewTol       = std::pow(eps, RealScalar{0.25f});
            RealScalar orthError     = std::numeric_limits<RealScalar>::quiet_NaN();
            RealScalar symmError     = std::numeric_limits<RealScalar>::quiet_NaN();
            RealScalar skewError     = std::numeric_limits<RealScalar>::quiet_NaN();
            RealScalar skewError_fwd = std::numeric_limits<RealScalar>::quiet_NaN();
            RealScalar skewError_adj = std::numeric_limits<RealScalar>::quiet_NaN();
            VectorReal proj_sum_a;
            VectorReal proj_sum_b;
            VectorReal scale_log;
            VectorIdxT mask;
            bool       force_refresh_a           = false;
            bool       refresh_by                = true;
            bool       gram_is_positive_definite = false;
            MaskPolicy maskPolicy                = MaskPolicy::RANDOMIZE;

            void analyze_l2_orthonormality(const Eigen::Ref<const MatrixType> &Y);
            void analyze_b_orthonormality(const Eigen::Ref<const MatrixType> &Y, const Eigen::Ref<const MatrixType> &B_Y);
            void analyze_bm_orthonormality(const Eigen::Ref<const MatrixType> &Y, const Eigen::Ref<const MatrixType> &B_Y);
            void analyze_l2_orthogonality(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &Y);
            void analyze_bm_orthogonality(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &B_X, const Eigen::Ref<const MatrixType> &Y,
                                          const Eigen::Ref<const MatrixType> &B_Y);
        };

        struct Status {
            VectorReal                eigVal;
            VectorReal                oldVal;
            VectorReal                absDiff;
            VectorReal                relDiff;
            RealScalar                initVal          = std::numeric_limits<RealScalar>::quiet_NaN();
            RealScalar                condition        = RealScalar{1};
            RealScalar                sensitivity      = RealScalar{1};
            RealScalar                op_norm_estimate = RealScalar{1};
            RealScalar                gap              = std::numeric_limits<RealScalar>::infinity();
            std::vector<Eigen::Index> optIdx;
            Eigen::Index              iter                 = 0;
            Eigen::Index              iter_last_restart    = 0;
            Eigen::Index              num_iters_inner      = 0;
            Eigen::Index              num_iters_inner_prev = 0;
            Eigen::Index              num_matvecs_inner    = 0;
            Eigen::Index              num_jdops_inner      = 0;
            Eigen::Index              num_matvecs          = 0;
            Eigen::Index              num_matvecs_total    = 0;
            Eigen::Index              num_precond          = 0;
            Eigen::Index              num_precond_inner    = 0;
            Eigen::Index              num_precond_total    = 0;
            tid::ur                   time_elapsed;
            tid::ur                   time_matvecs;
            tid::ur                   time_matvecs_inner;
            tid::ur                   time_matvecs_total;
            tid::ur                   time_precond;
            tid::ur                   time_precond_inner;
            tid::ur                   time_precond_total;
            tid::ur                   time_jdops_inner;
            RealScalar                inner_error_last = RealScalar{0};
            RealScalar                inner_tol_last   = RealScalar{0};
            bool                      rNorm_below_rnormTol = false;
            bool                      rNorm_below_gap      = false;
            VectorReal                rNorms;
            VectorReal                rNorms_init;
            std::deque<VectorReal>    rNorms_history;
            std::deque<VectorReal>    eigVals_history;
            std::deque<Eigen::Index>  matvecs_history;
            size_t                    max_history_size        = 5;
            Eigen::Index              saturation_count_eigVal = 0;
            Eigen::Index              saturation_count_rNorm  = 0;
            Eigen::Index              saturation_count_max    = 10;
            std::vector<std::string>  stopMessage             = {};
            StopReason                stopReason              = StopReason::none;
            OptRitz                   ritz_internal           = OptRitz::NONE;
        };

        struct AutoResidualCorrectionState {
            ResidualCorrectionType    active               = ResidualCorrectionType::CHEAP_OLSEN;
            ResidualCorrectionType    step_method          = ResidualCorrectionType::CHEAP_OLSEN;
            Eigen::Index              dwell                = 0;
            Eigen::Index              jd_steps_since_probe = 0;
            double                    step_time_start      = 0.0;
            std::vector<Eigen::Index> cheap_to_jd_switch_iters;
            std::vector<Eigen::Index> jd_to_cheap_switch_iters;
        };

        struct AutoSaturationMetric {
            bool         enabled        = false;
            bool         enough_history = false;
            bool         saturated      = false;
            Eigen::Index history_size   = 0;
            RealScalar   value          = RealScalar{0};
            RealScalar   stddev         = std::numeric_limits<RealScalar>::infinity();
            RealScalar   scale          = RealScalar{1};
            RealScalar   ratio          = std::numeric_limits<RealScalar>::infinity();
            RealScalar   threshold      = RealScalar{0};
        };

        struct AutoSaturationStatus {
            AutoSaturationMetric eigval;
            AutoSaturationMetric rnorm;
            bool                 ready = false;
        };

        protected:
        spdlog::level::level_enum logLevel = spdlog::level::warn;
        Logger::LoggerHandle      eiglog;
        tid::ur                   last_log_time = tid::ur();

        Eigen::Index qBlocks = 0;

        public:
        virtual ~base() = default;

        void setLogger(spdlog::level::level_enum logLevel, const std::string &name = "");

        base(Eigen::Index nev, Eigen::Index ncv, OptRitz ritz, const MatrixType &V, MatVec<Scalar> &A,
             spdlog::level::level_enum logLevel_ = spdlog::level::warn);

        Status       status = {};
        Eigen::Index N;
        Eigen::Index size;
        Eigen::Index nev        = 1; /*!< Number of requested eigenpairs. */
        Eigen::Index ncv        = 8; /*!< Maximum search-space columns. */
        Eigen::Index block_size = 2; /*!< Number of vectors in each solver block. */

        bool                             use_refined_rayleigh_ritz              = false; /*!< Use refined Rayleigh-Ritz extraction. */
        bool                             use_rayleigh_quotients_instead_of_evals = false; /*!< Replace projected eigenvalues with Rayleigh quotient estimates. */
        bool                             use_relative_rnorm_tolerance           = false; /*!< Enable tol_rnorm_relative convergence. */
        bool                             use_adaptive_inner_tolerance           = true;  /*!< Adapt inner_tol from previous inner-solver work. */
        bool                             use_b_inner_product                    = false; /*!< Use the B-metric inner product in generalized problems. */
        bool                             use_jd_b_only                          = false; /*!< Use the B-only generalized JD correction path. */
        bool                             use_krylov_schur_gdplusk_restart       = false; /*!< Use Krylov-Schur-style GD+k restart retention. */
        bool                             dev_append_extra_blocks_to_basis       = false; /*!< Development option to retain extra candidate blocks. */
        bool                             dev_skipjcb                            = false; /*!< Development option to skip Jacobi-Davidson correction blocks. */
        int                              chebyshev_filter_degree                = 0;     /*!< Degree of the optional Chebyshev filter. */
        ResidualCorrectionType           residual_correction_type         = ResidualCorrectionType::NONE;
        ResidualCorrectionType           residual_correction_type_internal = ResidualCorrectionType::NONE;
        OptRitz                          ritz;
        MatVec<Scalar>                  &A;
        MatrixType                       T;
        MatrixType                       Aproj, Bproj, W, Q;
        MatrixType                       AQ, BQ;
        MatrixType                       V;
        MatrixType                       AV;
        MatrixType                       BV;
        MatrixType                       V_prev;
        MatrixType                       K, K_prev;
        MatrixType                       S, S1, S2;
        MatrixType                       D;
        MatrixType                       M, AM, BM;
        VectorReal                       T_evals;
        MatrixType                       T1, T2, T_evecs;
        Eigen::HouseholderQR<MatrixType> hhqr;

        RealScalar   tol                                = eps * 10000; /*!< Absolute residual-norm convergence tolerance. */
        RealScalar   tol_rnorm_relative                 = 0;           /*!< Relative residual-norm convergence tolerance; zero disables it. */
        RealScalar   skewTol                           = std::sqrt(eps) * 10000;
        RealScalar   normTol                           = eps * 10;
        RealScalar   orthTol                           = eps * 100;
        RealScalar   quotTolB                          = RealScalar{1e-10f};
        Eigen::Index max_iters                         = 100l;         /*!< Maximum outer solver iterations; negative means unlimited. */
        Eigen::Index max_matvecs                       = -1l;          /*!< Maximum total matrix-vector products; negative means unlimited. */
        RealScalar   sat_eigval_threshold              = RealScalar{0}; /*!< Eigenvalue saturation threshold for stopping; zero disables this stop. */
        RealScalar   sat_rnorm_threshold               = RealScalar{0}; /*!< Residual-norm saturation threshold for stopping; zero disables this stop. */
        RealScalar   rnormRelDiffTol                   = std::numeric_limits<RealScalar>::epsilon();
        RealScalar   absDiffTol                        = std::numeric_limits<RealScalar>::epsilon() * 10000;
        RealScalar   relDiffTol                        = std::numeric_limits<RealScalar>::epsilon() * 10000;
        RealScalar   inner_tol                         = RealScalar{0.1f};  /*!< Target residual reduction for the inner correction solver. */
        Eigen::Index inner_max_iters                   = 1000;              /*!< Maximum iterations for the inner correction solver. */
        Eigen::Index auto_min_dwell_iters              = 10;                /*!< Minimum consecutive cheap-Olsen AUTO steps before JD activation may occur. */
        RealScalar   auto_sat_eigval_threshold         = RealScalar{1e-5f}; /*!< Eigenvalue saturation threshold for AUTO JD activation. */
        RealScalar   auto_sat_rnorm_threshold          = RealScalar{1e-2f}; /*!< Residual-norm saturation threshold for AUTO JD activation. */
        RealScalar   auto_jd_start_rnorm_threshold     = RealScalar{1e-5f}; /*!< Residual norm below which AUTO may activate JD; zero disables it. */
        Eigen::Index auto_cheap_probe_interval         = 5;                 /*!< JD steps before AUTO forces a cheap-Olsen probe. */
        RealScalar   auto_cheap_probe_factor           = RealScalar{1.0f};  /*!< Cheap probe must improve the Ritz value by this factor times max(rnorm^2, roundoff scale). */
        Eigen::Index maxPrevBlocks                     = 1;
        std::string  tag;
        AutoResidualCorrectionState auto_residual_correction;

        static std::string_view       ResidualCorrectionToString(ResidualCorrectionType rct);
        static ResidualCorrectionType StringToResidualCorrection(std::string_view rct);

        virtual std::string_view form_name() const                                  = 0;
        virtual bool             is_generalized_problem() const                     = 0;
        virtual MatrixType       MultB(const Eigen::Ref<const MatrixType> &X)       = 0;
        virtual MatrixType       MultB_inner(const Eigen::Ref<const MatrixType> &X) = 0;

        std::pair<MatrixType, VectorReal> get_residuals(const Eigen::Ref<VectorReal> &Y, const Eigen::Ref<MatrixType> &AV, const Eigen::Ref<MatrixType> &BV);
        RealScalar                        rNormTol(Eigen::Index n) const;
        VectorReal                        rNormTols() const;
        RealScalar                        get_rNorms_log10_change_per_matvec();
        RealScalar                        get_op_norm_estimate(std::optional<RealScalar> eigval = std::nullopt) const;

        MatrixType                MultA(const Eigen::Ref<const MatrixType> &X);
        MatrixType                MultP(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const VectorReal> &evals);
        std::vector<Eigen::Index> get_ritz_indices(OptRitz ritz, Eigen::Index offset, Eigen::Index num, const VectorReal &evals) const;

        void         init();
        virtual void build()        = 0;
        virtual void diagonalizeT() = 0;

        template<typename Comp>
        std::vector<Eigen::Index> getIndices(const VectorReal &v, const Eigen::Index offset, const Eigen::Index num, Comp comp) const {
            std::vector<Eigen::Index> idx(static_cast<size_t>(v.size()));
            Eigen::Index              numSort = std::min<Eigen::Index>(offset + num, v.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + numSort, idx.end(), [&](auto i, auto j) { return comp(v(i), v(j)); });
            return std::vector(idx.begin() + offset, idx.begin() + numSort);
        }

        void extractRitzVectors();
        void extractRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &AV, MatrixType &S, VectorReal &rNorms);
        void extractRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &AV, MatrixType &BV, MatrixType &S, VectorReal &rNorms);
        void orthonormalize_Z(Eigen::Ref<MatrixType> Z, const Eigen::Ref<const MatrixType> &T2);
        MatrixType get_refined_ritz_eigenvectors_std(const Eigen::Ref<const MatrixType> &Z, const Eigen::Ref<const VectorReal> &Y, const MatrixType &Q,
                                                     const MatrixType &AQ);
        MatrixType get_refined_ritz_eigenvectors_gen(const Eigen::Ref<const MatrixType> &Z, const Eigen::Ref<const VectorReal> &Y, const MatrixType &AQ,
                                                     const MatrixType &BQ);
        std::pair<MatrixType, MatrixType> get_bm_normalizer_for_the_projected_pencil(const MatrixType &T2);
        MatrixType get_optimal_rayleigh_ritz_matrix(const MatrixType &Z_rr, const MatrixType &Z_ref, const MatrixType &T1, const MatrixType &T2);
        void refinedRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &AQ, MatrixType &BQ, MatrixType &S, VectorReal &rNorms);
        void refinedRitzVectors(const std::vector<Eigen::Index> &optIdx, MatrixType &V, MatrixType &AQ, MatrixType &S, VectorReal &rNorms);
        void refinedRitzVectors();
        void preamble();
        void adjust_inner_tolerance(const Eigen::Ref<const MatrixType> &S);
        void updateStatus();
        void printStatus();
        result_view<Scalar> result() const;
        void                clear_result();

        void step();
        void run();

        void                   adjust_residual_correction_type();
        void                   update_auto_residual_correction_state();
        RealScalar             get_auto_rnorm_scalar(const VectorReal &rnorms) const;
        RealScalar             get_auto_probe_eigval_improvement() const;
        AutoSaturationMetric   get_auto_eigval_saturation_metric();
        AutoSaturationMetric   get_auto_rnorm_saturation_metric();
        AutoSaturationStatus   get_auto_saturation_status();
        MatrixType             cheap_Olsen_correction(const MatrixType &V, const MatrixType &S);
        MatrixType             full_Olsen_correction(const MatrixType &V, const MatrixType &S);
        MatrixType             jacobi_davidson_l2_correction(const MatrixType &V, const MatrixType &S, const VectorReal &evals);
        MatrixType             jacobi_davidson_bm_correction(const MatrixType &V, const MatrixType &BV, const MatrixType &S, const VectorReal &evals);
        MatrixType             get_sBlock(const MatrixType &S_in);

        void       set_maxPrevBlocks(Eigen::Index pb);
        void       assert_allFinite(const Eigen::Ref<const MatrixType> &X, const std::source_location &location = std::source_location::current());
        void       assert_l2_orthonormal(const Eigen::Ref<const MatrixType> &X, const OrthMeta &m,
                                         const std::source_location &location = std::source_location::current());
        void       assert_l2_orthogonal(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &Y, const OrthMeta &m,
                                        const std::source_location &location = std::source_location::current());
        void       assert_bm_orthonormal(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &B_X, const OrthMeta &m,
                                         const std::source_location &location = std::source_location::current());
        void       assert_bm_orthogonal(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &B_Y, const OrthMeta &m,
                                        const std::source_location &location = std::source_location::current());
        void       compress_cols(MatrixType &X, const VectorIdxT &mask);
        VectorReal get_standard_deviations(const std::deque<VectorReal> &v, bool apply_log10);
        VectorReal get_slopes(const std::deque<VectorReal> &v, bool apply_log10);
        bool       rNorms_have_saturated();
        bool       eigVals_have_saturated();
        bool       rNorms_saturated_for_auto_jd_start();
        bool       eigVals_saturated_for_auto_jd_start();
        bool       auto_jd_start_ready();
        void       block_l2_orthogonalize(const MatrixType &X, const MatrixType &AX, MatrixType &Y, MatrixType &AY, OrthMeta &m);
        void       block_l2_orthogonalize(const MatrixType &X, const MatrixType &AX, const MatrixType &BX, MatrixType &Y, MatrixType &AY, MatrixType &BY,
                                          OrthMeta &m);
        void       block_l2_orthonormalize(MatrixType &Y, MatrixType &AY, OrthMeta &m);
        void       block_l2_orthonormalize(MatrixType &Y, MatrixType &AY, MatrixType &BY, OrthMeta &m);
        void       block_bm_orthogonalize(const MatrixType &X, const MatrixType &AX, const MatrixType &BX, MatrixType &Y, MatrixType &AY, MatrixType &BY,
                                          OrthMeta &m);
        void       block_bm_orthonormalize(MatrixType &Y, MatrixType &AY, MatrixType &BY, OrthMeta &m);
    };
}

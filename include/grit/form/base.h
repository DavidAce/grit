#pragma once

#include "../enums.h"
#include "../internal/log.h"
#include "../internal/tid.h"
#include "../MatVec.h"
#include "../result.h"
#include <algorithm>
#include <cmath>
#include <deque>
#include <Eigen/Eigenvalues>
#include <fmt/format.h>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <source_location>
#include <string>
#include <string_view>
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
            void analyze_l2_orthogonality(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const MatrixType> &Y);
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
            Eigen::Index              iter              = 0;
            Eigen::Index              num_iters_inner   = 0;
            Eigen::Index              num_matvecs       = 0;
            Eigen::Index              num_matvecs_total = 0;
            Eigen::Index              num_precond       = 0;
            Eigen::Index              num_precond_total = 0;
            tid::ur                   time_elapsed;
            tid::ur                   time_matvecs;
            tid::ur                   time_precond;
            tid::ur                   time_matvecs_inner;
            tid::ur                   time_precond_inner;
            tid::ur                   time_jdops_inner;
            VectorReal                rNorms;
            VectorReal                rNorms_init;
            std::vector<std::string>  stopMessage   = {};
            StopReason                stopReason    = StopReason::none;
            OptRitz                   ritz_internal = OptRitz::NONE;
        };

        protected:
        spdlog::level::level_enum logLevel = spdlog::level::warn;
        Logger::LoggerHandle      eiglog;
        tid::ur                   last_log_time = tid::ur();

        Eigen::Index qBlocks = 0;

        virtual void set_form_jcbMaxBlockSize(Eigen::Index jcbMaxBlockSize);
        virtual void set_form_jcbOverlapSize(Eigen::Index jcbOverlapSize);
        virtual void set_form_jcbNumPasses(Eigen::Index jcbNumPasses);
        virtual void set_form_preconditioner_type(Preconditioner preconditioner_type);
        virtual void set_form_preconditioner_params(Eigen::Index maxiters, RealScalar initialTol, Eigen::Index jcbMaxBlockSize);

        public:
        virtual ~base() = default;

        void setLogger(spdlog::level::level_enum logLevel, const std::string &name = "");

        base(Eigen::Index nev, Eigen::Index ncv, OptRitz ritz, const MatrixType &V, MatVec<Scalar> &A,
             spdlog::level::level_enum logLevel_ = spdlog::level::warn);

        Status       status = {};
        Eigen::Index N;
        Eigen::Index size;
        Eigen::Index nev = 1;
        Eigen::Index ncv = 8;
        Eigen::Index b   = 2;

        bool                             use_preconditioner                           = false;
        bool                             use_refined_rayleigh_ritz                    = false;
        bool                             use_relative_rnorm_tolerance                 = false;
        bool                             use_adaptive_inner_tolerance                 = true;
        bool                             use_b_inner_product                          = false;
        bool                             use_krylov_schur_gdplusk_restart             = false;
        bool                             dev_append_extra_blocks_to_basis             = false;
        bool                             dev_orthogonalization_before_preconditioning = false;
        bool                             dev_skipjcb                                  = false;
        int                              chebyshev_filter_degree                      = 0;
        ResidualCorrectionType           residual_correction_type                     = ResidualCorrectionType::NONE;
        ResidualCorrectionType           residual_correction_type_internal            = ResidualCorrectionType::NONE;
        Preconditioner                   preconditioner_type                          = Preconditioner::NONE;
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

        RealScalar   abstol          = eps * 10000;
        RealScalar   reltol          = 0;
        RealScalar   skewTol         = std::sqrt(eps) * 10000;
        RealScalar   normTol         = eps * 10;
        RealScalar   orthTol         = eps * 100;
        RealScalar   quotTolB        = RealScalar{1e-10f};
        Eigen::Index max_iters       = 100l;
        Eigen::Index max_matvecs     = -1l;
        RealScalar   rnormRelDiffTol = std::numeric_limits<RealScalar>::epsilon();
        RealScalar   absDiffTol      = std::numeric_limits<RealScalar>::epsilon() * 10000;
        RealScalar   relDiffTol      = std::numeric_limits<RealScalar>::epsilon() * 10000;
        Eigen::Index maxPrevBlocks   = 1;
        std::string  tag;

        static std::string_view       ResidualCorrectionToString(ResidualCorrectionType rct);
        static ResidualCorrectionType StringToResidualCorrection(std::string_view rct);

        virtual std::string_view form_name() const                            = 0;
        virtual MatrixType       MultB(const Eigen::Ref<const MatrixType> &X) = 0;

        std::pair<MatrixType, VectorReal> get_residuals(const Eigen::Ref<VectorReal> &Y, const Eigen::Ref<MatrixType> &AV, const Eigen::Ref<MatrixType> &BV);
        RealScalar                        rNormTol(Eigen::Index n) const;
        VectorReal                        rNormTols() const;

        Eigen::Index get_jcbMaxBlockSize() const;
        Eigen::Index get_jcbOverlapSize() const;
        Eigen::Index get_jcbNumPasses() const;
        void         set_jcbMaxBlockSize(Eigen::Index jcbMaxBlockSize);
        void         set_jcbOverlapSize(Eigen::Index jcbOverlapSize);
        void         set_jcbNumPasses(Eigen::Index jcbNumPasses);
        void         set_preconditioner_type(Preconditioner preconditioner_type_);
        void         set_preconditioner_params(Eigen::Index maxiters = 1000, RealScalar initialTol = RealScalar{0.25f},
                                               Eigen::Index jcbMaxBlockSize = Eigen::Index{-1});

        MatrixType MultA(const Eigen::Ref<const MatrixType> &X);
        MatrixType MultP(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const VectorReal> &evals,
                         std::optional<const Eigen::Ref<const MatrixType>> initialGuess = std::nullopt);

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

        void                extractRitzVectors();
        void                preamble();
        void                updateStatus();
        void                printStatus();
        result_view<Scalar> result() const;
        void                clear_result();

        void step();
        void run();

        void set_maxPrevBlocks(Eigen::Index pb);
        void assert_allFinite(const Eigen::Ref<const MatrixType> &X, const std::source_location &location = std::source_location::current());
        void block_l2_orthonormalize(MatrixType &Y, MatrixType &AY, OrthMeta &m);
        void block_l2_orthonormalize(MatrixType &Y, MatrixType &AY, MatrixType &BY, OrthMeta &m);
    };
}

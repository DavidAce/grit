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
    class gdplusk : public form::base<Scalar_, form_> {
        public:
        using Base                   = form::base<Scalar_, form_>;
        using Scalar                 = typename Base::Scalar;
        using RealScalar             = typename Base::RealScalar;
        using MatrixType             = typename Base::MatrixType;
        using VectorType             = typename Base::VectorType;
        using VectorReal             = typename Base::VectorReal;
        using VectorIdxT             = typename Base::VectorIdxT;
        using fMultP_t               = typename Base::fMultP_t;
        using OrthMeta               = typename Base::OrthMeta;
        using BaseConfig             = typename Base::BaseConfig;
        using ResidualCorrectionType = typename Base::ResidualCorrectionType;
        using AutoSaturationInfo     = typename Base::AutoSaturationInfo;
        using AutoSaturationStatus   = typename Base::AutoSaturationStatus;

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
        using Base::relative_rNorms;
        using Base::relative_rNormScales;
        using Base::residual_correction_type_internal;
        using Base::status;
        using Base::T1;
        using Base::T2;
        using Base::T_evals;
        using Base::T_evecs;

        using Base::assert_allFinite;
        using Base::auto_residual_correction;
        using Base::block_bm_orthogonalize;
        using Base::block_bm_orthonormalize;
        using Base::block_l2_orthogonalize;
        using Base::block_l2_orthonormalize;
        using Base::cfg;
        using Base::clear_result;
        using Base::eiglog;
        using Base::eps;
        using Base::get_bm_normalizer_for_the_projected_pencil;
        using Base::get_op_norm_estimate;
        using Base::get_optimal_rayleigh_ritz_matrix;
        using Base::get_refined_ritz_eigenvectors_gen;
        using Base::get_refined_ritz_eigenvectors_std;
        using Base::get_standard_deviations;

        struct Config : BaseConfig {
            static constexpr RealScalar                         eps                              = std::numeric_limits<RealScalar>::epsilon();
            ResidualCorrectionType                              residual_correction_type         = ResidualCorrectionType::NONE;
            bool                                                use_adaptive_inner_tolerance     = false;
            bool                                                use_jd_b_only                    = false;
            bool                                                use_krylov_schur_gdplusk_restart = false;
            bool                                                inject_randomness                = false;
            Eigen::Index                                        max_basis_blocks                 = 8;
            Eigen::Index                                        maxRetainBlocks                  = 1;
            RealScalar                                          inner_tol                        = RealScalar{0.1f};
            Eigen::Index                                        inner_max_iters                  = 1000;
            Eigen::Index                                        auto_min_dwell_iters             = 10;
            RealScalar                                          auto_sat_eigval_threshold        = RealScalar{1e-3f};
            RealScalar                                          auto_sat_rnorm_threshold         = RealScalar{1e-2f};
            RealScalar                                          auto_jd_start_rnorm_threshold    = RealScalar{1e-5f};
            Eigen::Index                                        auto_cheap_probe_interval        = 5;
            RealScalar                                          auto_cheap_probe_factor          = RealScalar{1.0f};
            std::function<void(const gdplusk<Scalar, form_> &)> user_callback;
            Eigen::Index                                        max_extra_ritz_history    = 1;
            Eigen::Index                                        max_ritz_residual_history = 1;
        };

        Config config;

        gdplusk(Matvec<Scalar> &A) requires(form_ == grit::Form::STANDARD);
        gdplusk(Matvec<Scalar> &A, Matvec<Scalar> &B) requires(form_ == grit::Form::GENERALIZED);

        void               set_initial_guess(const MatrixType &V);
        void               clear_initial_guess();
        [[nodiscard]] bool has_initial_guess() const;
        void               run();

        private:
        Eigen::Index      max_mBlocks       = 1;
        Eigen::Index      max_sBlocks       = 1;
        RealScalar        current_inner_tol = RealScalar{0.1f};
        Eigen::Index      vBlocks           = 0;
        Eigen::Index      mBlocks           = 0;
        Eigen::Index      sBlocks           = 0;
        Eigen::Index      kBlocks           = 0;
        MatrixType        Q_new, AQ_new, BQ_new;
        MatrixType        G;
        const MatrixType *initial_guess_ptr = nullptr;
        MatrixType        empty_initial_guess;

        void shift_blocks_right(Eigen::Ref<MatrixType> matrix, Eigen::Index offset_old, Eigen::Index offset_new, Eigen::Index extent);
        void roll_blocks_left(Eigen::Ref<MatrixType> matrix, Eigen::Index offset, Eigen::Index extent);
        void selective_orthonormalize(const Eigen::Ref<const MatrixType> X, Eigen::Ref<MatrixType> Y, RealScalar breakdownTol, VectorIdxT &mask);
        void make_new_Q_block();
        void assert_config() const;
        [[nodiscard]] const MatrixType        &initial_guess() const;
        [[nodiscard]] static const MatrixType &default_initial_guess();
        void                                   assert_operator_config() const requires(form_ == grit::Form::STANDARD);
        void                                   assert_operator_config() const requires(form_ == grit::Form::GENERALIZED);
        static std::string_view                ResidualCorrectionToString(ResidualCorrectionType rct);
        static ResidualCorrectionType          StringToResidualCorrection(std::string_view rct);
        void                                   preamble() final;
        void                                   updateStatus() final;
        void                                   extractRitzVectors() final;
        void                                   run_user_callback() final;

        public:
        void                               adjust_inner_tolerance(const Eigen::Ref<const MatrixType> &S);
        void                               adjust_residual_correction_type();
        void                               update_auto_residual_correction_state();
        [[nodiscard]] RealScalar           get_auto_rnorm_scalar(const VectorReal &rnorms) const;
        [[nodiscard]] RealScalar           get_auto_probe_eigval_improvement() const;
        [[nodiscard]] AutoSaturationInfo   get_auto_eigval_saturation_info();
        [[nodiscard]] AutoSaturationInfo   get_auto_rnorm_saturation_info();
        [[nodiscard]] AutoSaturationStatus get_auto_saturation_status();
        [[nodiscard]] bool                 auto_jd_start_ready();
        [[nodiscard]] MatrixType           cheap_Olsen_correction(const MatrixType &V, const MatrixType &S);
        [[nodiscard]] MatrixType           full_Olsen_correction(const MatrixType &V, const MatrixType &S);
        [[nodiscard]] MatrixType           jacobi_davidson_l2_correction(const MatrixType &V, const MatrixType &S, const VectorReal &evals);
        [[nodiscard]] MatrixType jacobi_davidson_bm_correction(const MatrixType &V, const MatrixType &BV, const MatrixType &S, const VectorReal &evals)
            requires(form_ == grit::Form::GENERALIZED);
        [[nodiscard]] MatrixType get_sBlock(const MatrixType &S_in);

        public:
        void build() final;
        void build(MatrixType &Q, MatrixType &AQ, const MatrixType &Q_new, const MatrixType &AQ_new);
        void build(MatrixType &Q, MatrixType &AQ, MatrixType &BQ, const MatrixType &Q_new, const MatrixType &AQ_new, const MatrixType &BQ_new);
    };
}

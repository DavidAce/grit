#pragma once

#include <Eigen/Core>
#include <grit/config.h>
#include <grit/form/base.h>
#include <grit/prob/generalized.h>
#include <grit/prob/standard.h>
#include <type_traits>

namespace grit::form {
    template<typename Scalar>
    class standard;
    template<typename Scalar>
    class generalized;
}

namespace grit::algo {
    template<typename Form>
    class gdplusk : public Form {
        public:
        using Form::Form;
        using Scalar     = typename Form::Scalar;
        using RealScalar = typename Form::RealScalar;
        using MatrixType = typename Form::MatrixType;
        using VectorType = typename Form::VectorType;
        using VectorReal = typename Form::VectorReal;
        using VectorIdxT = typename Form::VectorIdxT;
        using fMultP_t   = typename Form::fMultP_t;
        using OrthMeta   = typename Form::OrthMeta;

        using Form::A;
        using Form::AQ;
        using Form::AV;
        using Form::block_bm_orthogonalize;
        using Form::block_bm_orthonormalize;
        using Form::block_l2_orthogonalize;
        using Form::block_l2_orthonormalize;
        using Form::block_size;
        using Form::BQ;
        using Form::BV;
        using Form::eiglog;
        using Form::get_bm_normalizer_for_the_projected_pencil;
        using Form::get_optimal_rayleigh_ritz_matrix;
        using Form::get_refined_ritz_eigenvectors_gen;
        using Form::get_refined_ritz_eigenvectors_std;
        using Form::get_sBlock;
        using Form::is_generalized_problem;
        using Form::K_prev;
        using Form::maxPrevBlocks;
        using Form::MultA;
        using Form::MultB;
        using Form::N;
        using Form::ncv;
        using Form::nev;
        using Form::orthonormalize_Z;
        using Form::Q;
        using Form::qBlocks;
        using Form::ritz;
        using Form::S;
        using Form::status;
        using Form::T1;
        using Form::T2;
        using Form::T_evals;
        using Form::T_evecs;
        using Form::use_b_inner_product;
        using Form::V;

        gdplusk(const grit::standard::problem<Scalar> &problem, const gdplusk_config<Scalar> &cfg) requires std::is_same_v<Form, grit::form::standard<Scalar>>;

        gdplusk(const grit::generalized::problem<Scalar> &problem, const gdplusk_config<Scalar> &cfg)
            requires std::is_same_v<Form, grit::form::generalized<Scalar>>;

        private:
        gdplusk_config<Scalar> config;
        Eigen::Index           max_mBlocks = 1;
        Eigen::Index           max_sBlocks = 1;
        Eigen::Index           vBlocks     = 0;
        Eigen::Index           mBlocks     = 0;
        Eigen::Index           sBlocks     = 0;
        Eigen::Index           kBlocks     = 0;
        MatrixType             Q_new, AQ_new, BQ_new;
        MatrixType             G;

        void shift_blocks_right(Eigen::Ref<MatrixType> matrix, Eigen::Index offset_old, Eigen::Index offset_new, Eigen::Index extent);
        void roll_blocks_left(Eigen::Ref<MatrixType> matrix, Eigen::Index offset, Eigen::Index extent);
        void selective_orthonormalize(const Eigen::Ref<const MatrixType> X, Eigen::Ref<MatrixType> Y, RealScalar breakdownTol, VectorIdxT &mask);
        void make_new_Q_block();
        void initialize_config();

        public:
        bool inject_randomness = false;
        void build() final;
        void build(MatrixType &Q, MatrixType &AQ, const MatrixType &Q_new, const MatrixType &AQ_new);
        void build(MatrixType &Q, MatrixType &AQ, MatrixType &BQ, const MatrixType &Q_new, const MatrixType &AQ_new, const MatrixType &BQ_new);
        void set_maxExtraRitzHistory(Eigen::Index m);
        void set_maxRitzResidualHistory(Eigen::Index s);
        void set_maxBasisBlocks(Eigen::Index bb);
        void set_maxRetainBlocks(Eigen::Index rb);
    };
}

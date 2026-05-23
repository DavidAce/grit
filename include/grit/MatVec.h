#pragma once

#include "internal/precondition/IterativeLinearSolverConfig.h"
#include "internal/scalars.h"
#include "internal/tid.h"
#include <complex>
#include <Eigen/Core>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

namespace grit {
    struct raw_t {};
    inline constexpr raw_t raw{};

    enum class Factorization { NONE, LLT, LDLT, LU, QR, ILUT, ILDLT };
    enum class Preconditioner { NONE, JACOBI, SOLVE };

    template<typename Scalar_>
    class MatVec {
        public:
        using Scalar     = Scalar_;
        using RealScalar = decltype(std::real(std::declval<Scalar>()));
        using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using VectorReal = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
        using MultAXFunc = std::function<MatrixType(const Eigen::Ref<const MatrixType> &)>;
        using RawMultAXFunc = std::function<void(const Scalar *, Scalar *, Eigen::Index, Eigen::Index)>;
        using MultPXFunc = std::function<MatrixType(const Eigen::Ref<const MatrixType> &, const Eigen::Ref<const VectorReal> &,
                                                    std::optional<const Eigen::Ref<const MatrixType>>)>;
        using CalcPcFunc = std::function<void(RealScalar)>;

        Factorization  factorization  = Factorization::NONE;
        Preconditioner preconditioner = Preconditioner::NONE;

        MatVec() = default;
        MatVec(Eigen::Index size_, MultAXFunc MultAX_);
        MatVec(MatVec &&other) noexcept;
        MatVec &operator=(MatVec &&other) noexcept;
        MatVec(const MatVec &other)            = delete;
        MatVec &operator=(const MatVec &other) = delete;
        ~MatVec();

        void set_MultAX(MultAXFunc MultAX_);
        void set_MultBX(MultAXFunc MultBX_);
        void set_MultPX(MultPXFunc MultPX_);
        void set_CalcPc(CalcPcFunc CalcPc_);
        void set_op_norm(RealScalar op_norm_);
        void set_size(Eigen::Index size_);

        [[nodiscard]] int          rows() const;
        [[nodiscard]] int          cols() const;
        [[nodiscard]] Eigen::Index get_size() const;
        [[nodiscard]] RealScalar   get_op_norm() const;

        MatrixType MultAX(const Eigen::Ref<const MatrixType> &X) const;

        MatrixType MultBX(const Eigen::Ref<const MatrixType> &X) const;

        MatrixType MultPX(const Eigen::Ref<const MatrixType> &X, const Eigen::Ref<const VectorReal> &evals);

        void CalcPc(RealScalar shift = RealScalar{0});
        void reset();

        void set_iterativeLinearSolverConfig(const IterativeLinearSolverConfig<Scalar> &cfg);
        void set_iterativeLinearSolverConfig(long maxiters = 1000, RealScalar tolerance = RealScalar{0.1f}, MatDef matdef = MatDef::IND);
        IterativeLinearSolverConfig<Scalar>       &get_iterativeLinearSolverConfig();
        const IterativeLinearSolverConfig<Scalar> &get_iterativeLinearSolverConfig() const;

        void               set_jcbMaxBlockSize(std::optional<long> jcbSize);
        void               set_jcbMaxBlockSize(long jcbSize);
        void               set_jcbOverlapSize(std::optional<long> size_);
        void               set_jcbOverlapSize(long size_);
        void               set_jcbNumPasses(std::optional<long> size_);
        void               set_jcbNumPasses(long size_);
        [[nodiscard]] long get_jcbMaxBlockSize() const;
        [[nodiscard]] long get_jcbOverlapSize() const;
        [[nodiscard]] long get_jcbNumPasses() const;

        mutable long             num_mv   = 0;
        mutable long             num_op   = 0;
        mutable long             num_pc   = 0;
        std::unique_ptr<tid::ur> t_multAx = std::make_unique<tid::ur>("Time MultAx");

        private:
        Eigen::Index                        size = 0;
        MultAXFunc                          MultAX_callback;
        MultAXFunc                          MultBX_callback;
        MultPXFunc                          MultPX_callback;
        CalcPcFunc                          CalcPc_callback;
        std::optional<RealScalar>           op_norm;
        IterativeLinearSolverConfig<Scalar> iLinSolvCfg;
        long                                jcbMaxBlockSize = -1;
        long                                jcbOverlapSize  = 0;
        long                                jcbNumPasses    = 1;
    };

    template<typename Scalar>
    MatVec<Scalar> matvec(Eigen::Index size, typename MatVec<Scalar>::MultAXFunc callback);

    template<typename Scalar>
    MatVec<Scalar> matvec(Eigen::Index size, raw_t, typename MatVec<Scalar>::RawMultAXFunc callback);
}

#pragma once

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

    template<typename Scalar_>
    class MatVec {
        public:
        using Scalar                   = Scalar_;
        using RealScalar               = decltype(std::real(std::declval<Scalar>()));
        using MatrixType               = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorType               = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using VectorReal               = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
        using MultAXFunc               = std::function<MatrixType(const Eigen::Ref<const MatrixType> &)>;
        using RawMultAXFunc            = std::function<void(const Scalar *, Scalar *, Eigen::Index, Eigen::Index)>;
        using PreconditionerUpdateFunc = std::function<void(RealScalar)>;
        using PreconditionerApplyFunc  = std::function<void(const Eigen::Ref<const VectorType> &, Eigen::Ref<VectorType>, RealScalar)>;

        MatVec() = default;
        MatVec(Eigen::Index size_, MultAXFunc MultAX_);
        MatVec(MatVec &&other) noexcept;
        MatVec &operator=(MatVec &&other) noexcept;
        MatVec(const MatVec &other)            = delete;
        MatVec &operator=(const MatVec &other) = delete;
        ~MatVec();

        void set_MultAX(MultAXFunc MultAX_);
        void set_MultBX(MultAXFunc MultBX_);
        void set_preconditioner_update(PreconditionerUpdateFunc preconditioner_update_);
        void set_preconditioner_apply(PreconditionerApplyFunc preconditioner_apply_);
        void set_op_norm(RealScalar op_norm_);
        void set_size(Eigen::Index size_);

        [[nodiscard]] int          rows() const;
        [[nodiscard]] int          cols() const;
        [[nodiscard]] Eigen::Index get_size() const;
        [[nodiscard]] RealScalar   get_op_norm() const;
        [[nodiscard]] bool         has_preconditioner_apply() const;

        MatrixType MultAX(const Eigen::Ref<const MatrixType> &X) const;

        MatrixType MultBX(const Eigen::Ref<const MatrixType> &X) const;

        void preconditioner_update(RealScalar theta) const;
        void preconditioner_apply(const Eigen::Ref<const VectorType> &x, Eigen::Ref<VectorType> y, RealScalar theta) const;

        void reset();

        mutable long             num_mv   = 0;
        mutable long             num_op   = 0;
        mutable long             num_pc   = 0;
        std::unique_ptr<tid::ur> t_multAx = std::make_unique<tid::ur>("Time MultAx");
        std::unique_ptr<tid::ur> t_multPc = std::make_unique<tid::ur>("Time MultPc");

        private:
        Eigen::Index              size = 0;
        MultAXFunc                MultAX_callback;
        MultAXFunc                MultBX_callback;
        PreconditionerUpdateFunc  preconditioner_update_callback;
        PreconditionerApplyFunc   preconditioner_apply_callback;
        std::optional<RealScalar> op_norm;
    };

    template<typename Scalar>
    MatVec<Scalar> matvec(Eigen::Index size, typename MatVec<Scalar>::MultAXFunc callback);

    template<typename Scalar>
    MatVec<Scalar> matvec(Eigen::Index size, raw_t, typename MatVec<Scalar>::RawMultAXFunc callback);
}

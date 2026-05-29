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
    struct ptr_t {};
    inline constexpr ptr_t ptr{};

    template<typename Scalar_>
    class Matvec {
        public:
        using Scalar                   = Scalar_;
        using RealScalar               = decltype(std::real(std::declval<Scalar>()));
        using MatrixType               = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorType               = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using VectorReal               = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
        using MultFunc                 = std::function<MatrixType(const Eigen::Ref<const MatrixType> &)>;
        using PtrMultFunc              = std::function<void(const Scalar *, Scalar *, Eigen::Index, Eigen::Index)>;
        using PreconditionerUpdateFunc = std::function<void(RealScalar)>;
        using PreconditionerApplyFunc  = std::function<void(const Eigen::Ref<const VectorType> &, Eigen::Ref<VectorType>, RealScalar)>;

        Matvec() = default;
        Matvec(Eigen::Index size_, MultFunc mult_);
        Matvec(Matvec &&other) noexcept;
        Matvec &operator=(Matvec &&other) noexcept;
        Matvec(const Matvec &other)            = delete;
        Matvec &operator=(const Matvec &other) = delete;
        ~Matvec();

        void set_mult(MultFunc mult_);
        void set_preconditioner_update(PreconditionerUpdateFunc preconditioner_update_);
        void set_preconditioner_apply(PreconditionerApplyFunc preconditioner_apply_);
        void set_op_norm(RealScalar op_norm_);
        void set_size(Eigen::Index size_);

        [[nodiscard]] int          rows() const;
        [[nodiscard]] int          cols() const;
        [[nodiscard]] Eigen::Index get_size() const;
        [[nodiscard]] RealScalar   get_op_norm() const;
        [[nodiscard]] bool         has_mult() const;
        [[nodiscard]] bool         has_preconditioner_apply() const;

        MatrixType mult(const Eigen::Ref<const MatrixType> &X) const;

        void preconditioner_update(RealScalar theta) const;
        void preconditioner_apply(const Eigen::Ref<const VectorType> &x, Eigen::Ref<VectorType> y, RealScalar theta) const;

        void reset();

        mutable long             num_mv    = 0;
        mutable long             num_pc    = 0;
        std::unique_ptr<tid::ur> t_mult    = std::make_unique<tid::ur>("Time Mult");
        std::unique_ptr<tid::ur> t_precond = std::make_unique<tid::ur>("Time Preconditioner");

        private:
        Eigen::Index              size = 0;
        MultFunc                  mult_callback;
        PreconditionerUpdateFunc  preconditioner_update_callback;
        PreconditionerApplyFunc   preconditioner_apply_callback;
        std::optional<RealScalar> op_norm;
    };

    template<typename Scalar>
    Matvec<Scalar> matvec(Eigen::Index size, typename Matvec<Scalar>::MultFunc callback);

    template<typename Scalar>
    Matvec<Scalar> matvec(Eigen::Index size, ptr_t, typename Matvec<Scalar>::PtrMultFunc callback);
}

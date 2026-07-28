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
    /*! Tag for the pointer-based matrix-vector callback overload. */
    struct ptr_t {};
    /*! Tag value for the pointer-based matrix-vector callback overload. */
    inline constexpr ptr_t ptr{};

    /*! Matrix-free linear operator with optional preconditioner callbacks. */
    template<typename Scalar_>
    class Matvec {
        public:
        using Scalar                   = Scalar_;                                               /*!< Scalar type of the operator. */
        using RealScalar               = decltype(std::real(std::declval<Scalar>()));           /*!< Real scalar type used for norms and Ritz values. */
        using MatrixType               = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>; /*!< Dense block of one or more vectors. */
        using VectorType               = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;              /*!< Dense single vector. */
        using VectorReal               = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;          /*!< Real-valued vector. */
        using MultFunc                 = std::function<MatrixType(const Eigen::Ref<const MatrixType> &)>;           /*!< Callback for Y = A X. */
        using PtrMultFunc              = std::function<void(const Scalar *, Scalar *, Eigen::Index, Eigen::Index)>; /*!< Pointer callback for Y = A X. */
        using PreconditionerUpdateFunc = std::function<void(RealScalar)>; /*!< Callback called when the shift theta changes. */
        using PreconditionerApplyFunc =
            std::function<void(const Eigen::Ref<const VectorType> &, Eigen::Ref<VectorType>, RealScalar)>; /*!< Callback for y = P(theta) x. */

        Matvec() = default;
        /*!
         * Construct an operator using a block matrix-vector callback.
         * @param size_ Square operator size.
         * @param mult_ Callback returning A X for a block X.
         */
        Matvec(Eigen::Index size_, MultFunc mult_);
        Matvec(Matvec &&other) noexcept;
        Matvec &operator=(Matvec &&other) noexcept;
        Matvec(const Matvec &other)            = delete;
        Matvec &operator=(const Matvec &other) = delete;
        ~Matvec();

        /*!
         * Set the block matrix-vector callback.
         * @param mult_ Callback returning A X for a block X.
         */
        void set_mult(MultFunc mult_);
        /*!
         * Set the callback used to prepare the preconditioner at shift theta.
         * @param preconditioner_update_ Callback called with the current shift.
         */
        void set_preconditioner_update(PreconditionerUpdateFunc preconditioner_update_);
        /*!
         * Set the callback used to apply the preconditioner.
         * @param preconditioner_apply_ Callback writing y = P(theta) x.
         */
        void set_preconditioner_apply(PreconditionerApplyFunc preconditioner_apply_);
        /*!
         * Set a known operator norm estimate.
         * @param op_norm_ Operator norm estimate.
         */
        void set_op_norm(RealScalar op_norm_);
        /*!
         * Set the square operator size.
         * @param size_ Number of rows and columns.
         */
        void set_size(Eigen::Index size_);

        /*! Number of operator rows. */
        [[nodiscard]] int rows() const;
        /*! Number of operator columns. */
        [[nodiscard]] int cols() const;
        /*! Square operator size. */
        [[nodiscard]] Eigen::Index get_size() const;
        /*! Operator norm estimate, or a fallback if none was supplied. */
        [[nodiscard]] RealScalar get_op_norm() const;
        /*! Whether a matrix-vector callback has been set. */
        [[nodiscard]] bool has_mult() const;
        /*! Whether a preconditioner apply callback has been set. */
        [[nodiscard]] bool has_preconditioner_apply() const;
        /*! Whether a preconditioner update callback has been set. */
        [[nodiscard]] bool has_preconditioner_update() const;

        /*!
         * Apply the operator to a block of vectors.
         * @param X Input block.
         * @return Output block A X.
         */
        MatrixType mult(const Eigen::Ref<const MatrixType> &X) const;

        /*!
         * Notify the preconditioner that the shift theta changed.
         * @param theta Current Ritz value or shift.
         */
        void preconditioner_update(RealScalar theta) const;
        /*!
         * Apply the preconditioner at shift theta.
         * @param x Input vector.
         * @param y Output vector.
         * @param theta Current Ritz value or shift.
         */
        void preconditioner_apply(const Eigen::Ref<const VectorType> &x, Eigen::Ref<VectorType> y, RealScalar theta) const;

        /*! Reset matrix-vector and preconditioner counters and timers. */
        void reset();

        mutable long             num_mv    = 0;                                                /*!< Number of matrix-vector callback calls. */
        mutable long             num_pc    = 0;                                                /*!< Number of preconditioner apply calls. */
        mutable long             num_pc_update    = 0;                                                /*!< Number of preconditioner update calls. */
        std::unique_ptr<tid::ur> t_mult    = std::make_unique<tid::ur>("Time Mult");           /*!< Timer for matrix-vector calls. */
        std::unique_ptr<tid::ur> t_precond        = std::make_unique<tid::ur>("Time Preconditioner"); /*!< Timer for preconditioner apply calls. */
        std::unique_ptr<tid::ur> t_precond_update = std::make_unique<tid::ur>("Time Preconditioner Update"); /*!< Timer for preconditioner update calls. */

        private:
        Eigen::Index              size = 0;
        MultFunc                  mult_callback;
        PreconditionerUpdateFunc  preconditioner_update_callback;
        PreconditionerApplyFunc   preconditioner_apply_callback;
        std::optional<RealScalar> op_norm;
    };

    /*!
     * Make a matrix-free operator from a block matrix-vector callback.
     * @param size Square operator size.
     * @param callback Callback returning A X for a block X.
     * @return Matrix-free operator wrapper.
     */
    template<typename Scalar>
    Matvec<Scalar> matvec(Eigen::Index size, typename Matvec<Scalar>::MultFunc callback);

    /*!
     * Make a matrix-free operator from a pointer-based matrix-vector callback.
     * @param size Square operator size.
     * @param callback Callback writing A X to the output pointer.
     * @return Matrix-free operator wrapper.
     */
    template<typename Scalar>
    Matvec<Scalar> matvec(Eigen::Index size, ptr_t, typename Matvec<Scalar>::PtrMultFunc callback);
}

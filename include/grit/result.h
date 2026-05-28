#pragma once

#include "enums.h"
#include <complex>
#include <Eigen/Core>
#include <string_view>
#include <utility>
#include <vector>

namespace grit::form {
    template<typename Scalar>
    class base;
}

namespace grit {
    template<typename Scalar_>
    class result_view {
        public:
        using Scalar     = Scalar_;
        using RealScalar = decltype(std::real(std::declval<Scalar>()));
        using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using VectorReal = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;

        result_view() = default;

        [[nodiscard]] const VectorReal                &eigVal() const;
        [[nodiscard]] const MatrixType                &eigVecs() const;
        [[nodiscard]] const VectorReal                &rNorms() const;
        [[nodiscard]] StopReason                       stopReason() const;
        [[nodiscard]] Eigen::Index                     iter() const;
        [[nodiscard]] Eigen::Index                     num_matvecs() const;
        [[nodiscard]] Eigen::Index                     num_iters_inner() const;
        [[nodiscard]] Eigen::Index                     num_matvecs_inner() const;
        [[nodiscard]] Eigen::Index                     num_jdops_inner() const;
        [[nodiscard]] Eigen::Index                     num_matvecs_total() const;
        [[nodiscard]] Eigen::Index                     num_precond() const;
        [[nodiscard]] Eigen::Index                     num_precond_inner() const;
        [[nodiscard]] Eigen::Index                     num_precond_total() const;
        [[nodiscard]] RealScalar                       seconds() const;
        [[nodiscard]] RealScalar                       inner_error_last() const;
        [[nodiscard]] RealScalar                       inner_tol_last() const;
        [[nodiscard]] Eigen::Index                     saturation_count_eigval() const;
        [[nodiscard]] Eigen::Index                     saturation_count_rnorm() const;
        [[nodiscard]] Eigen::Index                     saturation_count_max() const;
        [[nodiscard]] RealScalar                       op_norm_estimate() const;
        [[nodiscard]] RealScalar                       condition() const;
        [[nodiscard]] RealScalar                       sensitivity() const;
        [[nodiscard]] RealScalar                       gap() const;
        [[nodiscard]] bool                             rnorm_below_tol() const;
        [[nodiscard]] bool                             rnorm_below_gap() const;
        [[nodiscard]] std::string_view                 residual_correction_active_name() const;
        [[nodiscard]] std::string_view                 residual_correction_step_name() const;
        [[nodiscard]] Eigen::Index                     auto_dwell() const;
        [[nodiscard]] Eigen::Index                     auto_jd_steps_since_probe() const;
        [[nodiscard]] const std::vector<Eigen::Index> &cheap_to_jd_switch_iters() const;
        [[nodiscard]] const std::vector<Eigen::Index> &jd_to_cheap_switch_iters() const;

        private:
        friend class form::base<Scalar>;

        explicit result_view(const form::base<Scalar> &source_);

        const form::base<Scalar> *source = nullptr;
    };
}

#pragma once

#include <grit/form/base.h>
#include <grit/result.h>

namespace grit {
    template<typename Scalar_>
    result_view<Scalar_>::result_view(const form::base<Scalar> &source_) : source(&source_) {}

    template<typename Scalar_>
    const typename result_view<Scalar_>::VectorReal &result_view<Scalar_>::eigVal() const {
        return source->status.eigVal;
    }

    template<typename Scalar_>
    const typename result_view<Scalar_>::MatrixType &result_view<Scalar_>::eigVecs() const {
        return source->V;
    }

    template<typename Scalar_>
    const typename result_view<Scalar_>::VectorReal &result_view<Scalar_>::rNorms() const {
        return source->status.rNorms;
    }

    template<typename Scalar_>
    StopReason result_view<Scalar_>::stopReason() const {
        return source->status.stopReason;
    }

    template<typename Scalar_>
    Eigen::Index result_view<Scalar_>::iter() const {
        return source->status.iter;
    }

    template<typename Scalar_>
    Eigen::Index result_view<Scalar_>::num_matvecs() const {
        return source->status.num_matvecs;
    }

    template<typename Scalar_>
    Eigen::Index result_view<Scalar_>::num_iters_inner() const {
        return source->status.num_iters_inner;
    }

    template<typename Scalar_>
    Eigen::Index result_view<Scalar_>::num_matvecs_inner() const {
        return source->status.num_matvecs_inner;
    }

    template<typename Scalar_>
    Eigen::Index result_view<Scalar_>::num_jdops_inner() const {
        return source->status.num_jdops_inner;
    }

    template<typename Scalar_>
    Eigen::Index result_view<Scalar_>::num_matvecs_total() const {
        return source->status.num_matvecs_total;
    }

    template<typename Scalar_>
    Eigen::Index result_view<Scalar_>::num_precond() const {
        return source->status.num_precond;
    }

    template<typename Scalar_>
    Eigen::Index result_view<Scalar_>::num_precond_inner() const {
        return source->status.num_precond_inner;
    }

    template<typename Scalar_>
    Eigen::Index result_view<Scalar_>::num_precond_total() const {
        return source->status.num_precond_total;
    }

    template<typename Scalar_>
    typename result_view<Scalar_>::RealScalar result_view<Scalar_>::seconds() const {
        return static_cast<RealScalar>(source->status.time_elapsed.get_time());
    }

    template<typename Scalar_>
    typename result_view<Scalar_>::RealScalar result_view<Scalar_>::inner_error_last() const {
        return source->status.inner_error_last;
    }

    template<typename Scalar_>
    typename result_view<Scalar_>::RealScalar result_view<Scalar_>::inner_tol_last() const {
        return source->status.inner_tol_last;
    }

    template<typename Scalar_>
    Eigen::Index result_view<Scalar_>::saturation_count_eigval() const {
        return source->status.saturation_count_eigVal;
    }

    template<typename Scalar_>
    Eigen::Index result_view<Scalar_>::saturation_count_rnorm() const {
        return source->status.saturation_count_rNorm;
    }

    template<typename Scalar_>
    Eigen::Index result_view<Scalar_>::saturation_count_max() const {
        return source->status.saturation_count_max;
    }

    template<typename Scalar_>
    typename result_view<Scalar_>::RealScalar result_view<Scalar_>::op_norm_estimate() const {
        return source->status.op_norm_estimate;
    }

    template<typename Scalar_>
    typename result_view<Scalar_>::RealScalar result_view<Scalar_>::condition() const {
        return source->status.condition;
    }

    template<typename Scalar_>
    typename result_view<Scalar_>::RealScalar result_view<Scalar_>::sensitivity() const {
        return source->status.sensitivity;
    }

    template<typename Scalar_>
    typename result_view<Scalar_>::RealScalar result_view<Scalar_>::gap() const {
        return source->status.gap;
    }

    template<typename Scalar_>
    bool result_view<Scalar_>::rnorm_below_tol() const {
        return source->status.rNorm_below_rnormTol;
    }

    template<typename Scalar_>
    bool result_view<Scalar_>::rnorm_below_gap() const {
        return source->status.rNorm_below_gap;
    }

    template<typename Scalar_>
    std::string_view result_view<Scalar_>::residual_correction_active_name() const {
        return form::base<Scalar>::ResidualCorrectionToString(source->auto_residual_correction.active);
    }

    template<typename Scalar_>
    std::string_view result_view<Scalar_>::residual_correction_step_name() const {
        return form::base<Scalar>::ResidualCorrectionToString(source->auto_residual_correction.step_method);
    }

    template<typename Scalar_>
    Eigen::Index result_view<Scalar_>::auto_dwell() const {
        return source->auto_residual_correction.dwell;
    }

    template<typename Scalar_>
    Eigen::Index result_view<Scalar_>::auto_jd_steps_since_probe() const {
        return source->auto_residual_correction.jd_steps_since_probe;
    }

    template<typename Scalar_>
    const std::vector<Eigen::Index> &result_view<Scalar_>::cheap_to_jd_switch_iters() const {
        return source->auto_residual_correction.cheap_to_jd_switch_iters;
    }

    template<typename Scalar_>
    const std::vector<Eigen::Index> &result_view<Scalar_>::jd_to_cheap_switch_iters() const {
        return source->auto_residual_correction.jd_to_cheap_switch_iters;
    }
}

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
    Eigen::Index result_view<Scalar_>::num_precond_total() const {
        return source->status.num_precond_total;
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

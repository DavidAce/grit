#pragma once

#include <grit/form/base.h>
#include <grit/solver_view.h>

namespace grit {
    namespace internal {
        template<typename Variant, typename Fn>
        decltype(auto) visit_solver_source(const Variant &variant, Fn &&fn) {
            return std::visit([&](auto *src) -> decltype(auto) { return fn(*src); }, variant);
        }
    }

    template<typename Scalar_>
    template<grit::Form form>
    solver_view<Scalar_>::solver_view(const form::base<Scalar, form> &source_) : source(&source_) {}

    template<typename Scalar_>
    const typename solver_view<Scalar_>::VectorReal &solver_view<Scalar_>::eigVal() const {
        return internal::visit_solver_source(source, [](const auto &src) -> const VectorReal & { return src.status.eigVal; });
    }

    template<typename Scalar_>
    const typename solver_view<Scalar_>::MatrixType &solver_view<Scalar_>::eigVecs() const {
        return internal::visit_solver_source(source, [](const auto &src) -> const MatrixType & { return src.V; });
    }

    template<typename Scalar_>
    const typename solver_view<Scalar_>::VectorReal &solver_view<Scalar_>::rNorms() const {
        return internal::visit_solver_source(source, [](const auto &src) -> const VectorReal & { return src.status.rNorms; });
    }

    template<typename Scalar_>
    StopReason solver_view<Scalar_>::stopReason() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.stopReason; });
    }

    template<typename Scalar_>
    Eigen::Index solver_view<Scalar_>::iter() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.iter; });
    }

    template<typename Scalar_>
    Eigen::Index solver_view<Scalar_>::num_matvecs() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_matvecs; });
    }

    template<typename Scalar_>
    Eigen::Index solver_view<Scalar_>::num_iters_inner() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_iters_inner; });
    }

    template<typename Scalar_>
    Eigen::Index solver_view<Scalar_>::num_matvecs_inner() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_matvecs_inner; });
    }

    template<typename Scalar_>
    Eigen::Index solver_view<Scalar_>::num_jdops_inner() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_jdops_inner; });
    }

    template<typename Scalar_>
    Eigen::Index solver_view<Scalar_>::num_matvecs_total() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_matvecs_total; });
    }

    template<typename Scalar_>
    Eigen::Index solver_view<Scalar_>::num_precond() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_precond; });
    }

    template<typename Scalar_>
    Eigen::Index solver_view<Scalar_>::num_precond_inner() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_precond_inner; });
    }

    template<typename Scalar_>
    Eigen::Index solver_view<Scalar_>::num_precond_total() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_precond_total; });
    }

    template<typename Scalar_>
    typename solver_view<Scalar_>::RealScalar solver_view<Scalar_>::seconds() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_elapsed.get_time()); });
    }

    template<typename Scalar_>
    typename solver_view<Scalar_>::RealScalar solver_view<Scalar_>::inner_error_last() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.inner_error_last; });
    }

    template<typename Scalar_>
    typename solver_view<Scalar_>::RealScalar solver_view<Scalar_>::inner_tol_last() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.inner_tol_last; });
    }

    template<typename Scalar_>
    Eigen::Index solver_view<Scalar_>::saturation_count_eigval() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.saturation_count_eigVal; });
    }

    template<typename Scalar_>
    Eigen::Index solver_view<Scalar_>::saturation_count_rnorm() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.saturation_count_rNorm; });
    }

    template<typename Scalar_>
    Eigen::Index solver_view<Scalar_>::saturation_count_max() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.saturation_count_max; });
    }

    template<typename Scalar_>
    typename solver_view<Scalar_>::RealScalar solver_view<Scalar_>::op_norm_estimate() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.op_norm_estimate; });
    }

    template<typename Scalar_>
    typename solver_view<Scalar_>::RealScalar solver_view<Scalar_>::condition() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.condition; });
    }

    template<typename Scalar_>
    typename solver_view<Scalar_>::RealScalar solver_view<Scalar_>::sensitivity() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.sensitivity; });
    }

    template<typename Scalar_>
    typename solver_view<Scalar_>::RealScalar solver_view<Scalar_>::gap() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.gap; });
    }

    template<typename Scalar_>
    bool solver_view<Scalar_>::rnorm_below_tol() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.rNorm_below_rnormTol; });
    }

    template<typename Scalar_>
    bool solver_view<Scalar_>::rnorm_below_gap() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.rNorm_below_gap; });
    }

    template<typename Scalar_>
    std::string_view solver_view<Scalar_>::residual_correction_active_name() const {
        return internal::visit_solver_source(source, [](const auto &src) {
            return std::remove_cvref_t<decltype(src)>::ResidualCorrectionToString(src.auto_residual_correction.active);
        });
    }

    template<typename Scalar_>
    std::string_view solver_view<Scalar_>::residual_correction_step_name() const {
        return internal::visit_solver_source(source, [](const auto &src) {
            return std::remove_cvref_t<decltype(src)>::ResidualCorrectionToString(src.auto_residual_correction.step_method);
        });
    }

    template<typename Scalar_>
    Eigen::Index solver_view<Scalar_>::auto_dwell() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.auto_residual_correction.dwell; });
    }

    template<typename Scalar_>
    Eigen::Index solver_view<Scalar_>::auto_jd_steps_since_probe() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.auto_residual_correction.jd_steps_since_probe; });
    }

    template<typename Scalar_>
    const std::vector<Eigen::Index> &solver_view<Scalar_>::cheap_to_jd_switch_iters() const {
        return internal::visit_solver_source(source, [](const auto &src) -> const std::vector<Eigen::Index> & { return src.auto_residual_correction.cheap_to_jd_switch_iters; });
    }

    template<typename Scalar_>
    const std::vector<Eigen::Index> &solver_view<Scalar_>::jd_to_cheap_switch_iters() const {
        return internal::visit_solver_source(source, [](const auto &src) -> const std::vector<Eigen::Index> & { return src.auto_residual_correction.jd_to_cheap_switch_iters; });
    }
}

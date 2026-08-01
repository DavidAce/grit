#pragma once

#include "grit/form/base.h"
#include "grit/Result.h"

namespace grit {
    namespace internal {
        template<typename Variant, typename Fn>
        decltype(auto) visit_solver_source(const Variant &variant, Fn &&fn) {
            return std::visit([&](auto *src) -> decltype(auto) { return fn(*src); }, variant);
        }
    }

    namespace form {
        template<typename Scalar_, grit::Form form_>
        ResultView<Scalar_> base<Scalar_, form_>::get_result_view() const {
            return ResultView<Scalar_>(*this);
        }

        template<typename Scalar_, grit::Form form_>
        Result<Scalar_> base<Scalar_, form_>::get_result() const {
            return get_result_view().to_result();
        }
    }

    template<typename Scalar_>
    template<grit::Form form>
    ResultView<Scalar_>::ResultView(const form::base<Scalar, form> &source_) : source(&source_) {}

    template<typename Scalar_>
    const typename ResultView<Scalar_>::VectorReal &ResultView<Scalar_>::eigVal() const {
        return internal::visit_solver_source(source, [](const auto &src) -> const VectorReal & { return src.status.eigVal; });
    }

    template<typename Scalar_>
    const typename ResultView<Scalar_>::MatrixType &ResultView<Scalar_>::eigVecs() const {
        return internal::visit_solver_source(source, [](const auto &src) -> const MatrixType & { return src.V; });
    }

    template<typename Scalar_>
    const typename ResultView<Scalar_>::VectorReal &ResultView<Scalar_>::rNormsAbs() const {
        return internal::visit_solver_source(source, [](const auto &src) -> const VectorReal & { return src.status.rNormsAbs; });
    }

    template<typename Scalar_>
    StopReason ResultView<Scalar_>::stopReason() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.stopReason; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::outer_iter() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.outer_iter; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::num_matvecs() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_matvecs; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::num_inner_iters() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_inner_iters; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::num_matvecs_inner() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_matvecs_inner; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::num_matvecs_a() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_matvecs_a; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::num_matvecs_b() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_matvecs_b; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::num_matvecs_a_inner() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_matvecs_a_inner; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::num_matvecs_b_inner() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_matvecs_b_inner; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::num_matvecs_a_total() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_matvecs_a_total; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::num_matvecs_b_total() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_matvecs_b_total; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::num_operator_inner() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_operator_inner; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::num_matvecs_total() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_matvecs_total; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::num_precond() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_precond; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::num_precond_inner() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_precond_inner; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::num_precond_total() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_precond_total; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::num_preconditioner_updates() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_preconditioner_updates; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::num_preconditioner_updates_inner() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_preconditioner_updates_inner; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::num_preconditioner_updates_total() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_preconditioner_updates_total; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::num_preconditioner_apply_inner() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_preconditioner_apply_inner; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::num_preconditioner_apply_inner_total() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_preconditioner_apply_inner_total; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::num_preconditioner_apply_total() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.num_preconditioner_apply_total; });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_elapsed.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_solve_inner() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_solve_inner.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_operator_inner() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_operator_inner.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_matvecs() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_matvecs.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_matvecs_inner() const {
        return time_matvecs_a_inner() + time_matvecs_b_inner();
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_matvecs_a() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_matvecs_a.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_matvecs_b() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_matvecs_b.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_matvecs_a_inner() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_matvecs_a_inner.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_matvecs_b_inner() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_matvecs_b_inner.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_preconditioner_inner() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_preconditioner_inner.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_preconditioner_update() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_preconditioner_update.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_preconditioner_update_inner() const {
        return internal::visit_solver_source(source,
                                             [](const auto &src) { return static_cast<RealScalar>(src.status.time_preconditioner_update_inner.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_preconditioner_update_total() const {
        return internal::visit_solver_source(source, [](const auto &src) {
            return static_cast<RealScalar>(src.status.time_preconditioner_update.get_time() + src.status.time_preconditioner_update_inner.get_time());
        });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_preconditioner_apply_inner() const {
        return internal::visit_solver_source(source,
                                             [](const auto &src) { return static_cast<RealScalar>(src.status.time_preconditioner_apply_inner.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_preconditioner_apply_total() const {
        return internal::visit_solver_source(source, [](const auto &src) {
            return static_cast<RealScalar>(src.status.time_precond.get_time() + src.status.time_preconditioner_apply_inner.get_time());
        });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_project_left_inner() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_project_left_inner.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_project_right_inner() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_project_right_inner.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_residual_correction() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_residual_correction.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_build() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_build.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_status_update() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_status_update.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_orthogonalize() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_orthogonalize.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_orthonormalize() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_orthonormalize.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_orth_project() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_orth_project.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_orth_factor() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_orth_factor.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_orth_update() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_orth_update.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_orth_refresh() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_orth_refresh.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_orth_mask() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_orth_mask.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_diagonalize() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_diagonalize.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_extract_ritz() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_extract_ritz.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::time_restart() const {
        return internal::visit_solver_source(source, [](const auto &src) { return static_cast<RealScalar>(src.status.time_restart.get_time()); });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::inner_error_last() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.inner_error_last; });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::inner_tol_last() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.inner_tol_last; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::saturation_count_eigval() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.saturation_count_eigVal; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::saturation_count_rnorm() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.saturation_count_rNorm; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::saturation_count_max() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.saturation_count_max; });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::op_norm_estimate() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.op_norm_estimate; });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::condition() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.condition; });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::sensitivity() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.sensitivity; });
    }

    template<typename Scalar_>
    typename ResultView<Scalar_>::RealScalar ResultView<Scalar_>::gap() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.gap; });
    }

    template<typename Scalar_>
    bool ResultView<Scalar_>::residual_converged() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.residual_converged; });
    }

    template<typename Scalar_>
    bool ResultView<Scalar_>::residual_below_gap() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.status.residual_below_gap; });
    }

    template<typename Scalar_>
    std::string_view ResultView<Scalar_>::residual_correction_active_name() const {
        return internal::visit_solver_source(
            source, [](const auto &src) { return std::remove_cvref_t<decltype(src)>::ResidualCorrectionToString(src.auto_residual_correction.active); });
    }

    template<typename Scalar_>
    std::string_view ResultView<Scalar_>::residual_correction_iteration_name() const {
        return internal::visit_solver_source(source, [](const auto &src) {
            return std::remove_cvref_t<decltype(src)>::ResidualCorrectionToString(src.auto_residual_correction.iteration_method);
        });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::auto_cheap_olsen_iters() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.auto_residual_correction.cheap_olsen_iters; });
    }

    template<typename Scalar_>
    Eigen::Index ResultView<Scalar_>::auto_jd_outer_iters_since_probe() const {
        return internal::visit_solver_source(source, [](const auto &src) { return src.auto_residual_correction.jd_outer_iters_since_probe; });
    }

    template<typename Scalar_>
    const std::vector<Eigen::Index> &ResultView<Scalar_>::cheap_olsen_to_jd_switch_outer_iters() const {
        return internal::visit_solver_source(
            source, [](const auto &src) -> const std::vector<Eigen::Index> & { return src.auto_residual_correction.cheap_olsen_to_jd_switch_outer_iters; });
    }

    template<typename Scalar_>
    const std::vector<Eigen::Index> &ResultView<Scalar_>::jd_to_cheap_olsen_switch_outer_iters() const {
        return internal::visit_solver_source(
            source, [](const auto &src) -> const std::vector<Eigen::Index> & { return src.auto_residual_correction.jd_to_cheap_olsen_switch_outer_iters; });
    }
}

#include "report.h"
#include "memory.h"
#include <Eigen/Core>
#include <format>
#include <print>
#include <string>
#include <type_traits>
#include <vector>

namespace bench_standard {
    namespace {
        template<typename T>
        std::string list_text(const std::vector<T> &values) {
            std::string text = "[";
            for(std::size_t i = 0; i < values.size(); ++i) {
                if(i > 0) text += ",";
                text += std::format("{}", values[i]);
            }
            text += "]";
            return text;
        }

        std::string bool_list_text(const std::vector<bool> &values) {
            std::string text = "[";
            for(std::size_t i = 0; i < values.size(); ++i) {
                if(i > 0) text += ",";
                text += values[i] ? "true" : "false";
            }
            text += "]";
            return text;
        }

    }

    std::string_view bool_text(bool value) { return value ? "true" : "false"; }

    std::string_view residual_correction_name(ResidualCorrection correction) {
        switch(correction) {
            case ResidualCorrection::NONE: return "NONE";
            case ResidualCorrection::CHEAP_OLSEN: return "CHEAP_OLSEN";
            case ResidualCorrection::FULL_OLSEN: return "FULL_OLSEN";
            case ResidualCorrection::JACOBI_DAVIDSON: return "JACOBI_DAVIDSON";
            case ResidualCorrection::AUTO: return "AUTO";
        }
        return "NONE";
    }

    std::string limit_text(Eigen::Index value) { return value < 0 ? "unlimited" : std::format("{}", value); }

    std::string_view log_level_name(spdlog::level::level_enum level) {
        switch(level) {
            case spdlog::level::trace: return "trace";
            case spdlog::level::debug: return "debug";
            case spdlog::level::info: return "info";
            case spdlog::level::warn: return "warn";
            case spdlog::level::err: return "err";
            case spdlog::level::critical: return "critical";
            case spdlog::level::off: return "off";
            case spdlog::level::n_levels: return "n_levels";
        }
        return "warn";
    }

    void print_sweep_config(const CliOptions &opts, std::size_t cases) {
        std::println("sweep:");
        std::println("  algo: {}", algo_name(opts.algo));
        std::println("  cases: {} | reps: {} | seed: {} | seed per repetition: seed + rep - 1", cases, opts.reps, opts.seed);
        std::println("  limits: max-iters {} | max-matvecs {} | abstol {} | reltol {:.4e} | rescaled rnorm tolerance: {} | eigval saturation {:.4e} | "
                     "rnorm_rel saturation {:.4e}",
                     limit_text(opts.max_iters), limit_text(opts.max_matvecs), list_text(opts.abstol), opts.reltol,
                     bool_text(opts.use_rescaled_rnorm_tolerance), opts.sat_eigval_threshold, opts.sat_rnorm_threshold);
        std::println("  sweep axes: ncv {} | block-size {} | ritz {}", list_text(opts.ncv), list_text(opts.block_size), opts.ritz);
        std::println("  sweep axes: abstol {}", list_text(opts.abstol));
        std::println("  ritz stabilization tolerance: {:.3e}", opts.ritz_stabilization_tolerance);
        if(opts.algo == Algo::gdplusk) {
            std::println("  sweep axes: residual correction {} | inner tol {} | inner max iterations {} | refined {} | adaptive {}", opts.residual_correction,
                         list_text(opts.inner_tol), list_text(opts.inner_max_iters), bool_list_text(opts.use_refined_rayleigh_ritz),
                         bool_list_text(opts.use_adaptive_inner_tolerance));
            if(opts.residual_correction.find("auto") != std::string::npos || opts.residual_correction.find("AUTO") != std::string::npos) {
                std::println("  auto correction: probe interval {} | probe length {} | max probes {}", opts.auto_probe_interval, opts.auto_probe_length,
                             opts.auto_max_probes);
            }
        } else if(opts.algo == Algo::lanczos) {
            std::println("  sweep axes: max retain blocks {} | refined {}", list_text(opts.max_retain_blocks), bool_list_text(opts.use_refined_rayleigh_ritz));
        } else {
            std::println("  sweep axes: refined {}", bool_list_text(opts.use_refined_rayleigh_ritz));
        }
        std::println("  initial guess: {}", opts.initial_guess.empty() ? "random" : opts.initial_guess);
        if(!opts.save_eigvec.empty()) std::println("  save eigvec: {}", opts.save_eigvec);
        if(!opts.save_results.empty()) std::println("  save results: {}", opts.save_results);
        std::println("  Eigen threads: {} | log level: {}", Eigen::nbThreads(), log_level_name(opts.log_level));
        std::println("  matvec: Eigen row-major sparse * dense");
    }

    void print_result_header(Algo algo) {
        if(algo == Algo::gdplusk) {
            std::println("{:<5} {:<5} {:>5} {:>5} {:<4} {:<17} {:<7} {:<8} {:>10} {:>10} {:>7} {:<24} {:>18} {:>12} {:>12} {:>8} {:>8} {:>8} {:>8} {:>8} "
                         "{:>9} {:>12} {:>12} {:>12} {:>12}",
                         "case", "rep", "ncv", "blk", "ritz", "correction", "refined", "adaptive", "abstol", "inner_tol", "inner", "stop", "eigval",
                         "rnorm_abs", "rnorm_rel", "outer_iter", "matvec", "outer", "inner", "op_inner", "jd_switch", "time[s]", "VmRSS", "VmHWM", "VmPeak");
            return;
        }

        if(algo == Algo::lanczos) {
            std::println("{:<5} {:<5} {:<8} {:>5} {:>5} {:>6} {:<4} {:<7} {:>10} {:<24} {:>18} {:>12} {:>12} {:>8} {:>8} {:>8} {:>12} {:>12} {:>12}", "case",
                         "rep", "algo", "ncv", "blk", "retain", "ritz", "refined", "abstol", "stop", "eigval", "rnorm_abs", "rnorm_rel", "outer_iter", "matvec",
                         "outer", "time[s]", "VmRSS", "VmHWM");
            return;
        }

        std::println("{:<5} {:<5} {:<8} {:>5} {:>5} {:<4} {:<7} {:>10} {:<24} {:>18} {:>12} {:>12} {:>8} {:>8} {:>8} {:>12} {:>12} {:>12}", "case", "rep",
                     "algo", "ncv", "blk", "ritz", "refined", "abstol", "stop", "eigval", "rnorm_abs", "rnorm_rel", "outer_iter", "matvec", "outer", "time[s]",
                     "VmRSS", "VmHWM");
    }

    void print_result_row(const SolveResult &result) {
        std::visit(
            [&](const auto &typed_result) {
                const auto &snapshot = typed_result.final;
                using ResultType     = std::remove_cvref_t<decltype(typed_result)>;
                if constexpr(std::is_same_v<ResultType, GdpluskSolveResult>) {
                    const auto jd_switch =
                        snapshot.first_cheap_olsen_to_jd_outer_iter < 0 ? std::string{"n/a"} : std::format("{}", snapshot.first_cheap_olsen_to_jd_outer_iter);
                    std::println("{:<5} {:<5} {:>5} {:>5} {:<4} {:<17} {:<7} {:<8} {:>10.2e} {:>10.2e} {:>7} {:<24} {:>18.10e} {:>12.4e} {:>12.4e} {:>8} {:>8} "
                                 "{:>8} {:>8} {:>8} {:>9} {:>12.6f} {:>12} {:>12} {:>12}",
                                 snapshot.case_id, snapshot.rep, snapshot.ncv, snapshot.block_size, std::string_view(snapshot.ritz),
                                 std::string_view(snapshot.residual_correction), bool_text(snapshot.use_refined_rayleigh_ritz),
                                 bool_text(snapshot.use_adaptive_inner_tolerance), snapshot.abstol, snapshot.inner_tol, snapshot.inner_max_iters,
                                 std::string_view(snapshot.stop_reason), snapshot.eigenvalue, snapshot.rnorm_abs, snapshot.rnorm_rel, snapshot.outer_iterations,
                                 snapshot.matvecs, snapshot.outer_matvecs, snapshot.inner_matvecs, snapshot.operator_inner, jd_switch, snapshot.time,
                                 mem_size(snapshot.vmrss_mib), mem_size(snapshot.vmhwm_mib), mem_size(snapshot.vmpeak_mib));
                } else if constexpr(std::is_same_v<ResultType, LanczosSolveResult>) {
                    std::println("{:<5} {:<5} {:<8} {:>5} {:>5} {:>6} {:<4} {:<7} {:>10.2e} {:<24} {:>18.10e} {:>12.4e} {:>12.4e} {:>8} {:>8} {:>8} {:>12.6f} "
                                 "{:>12} {:>12}",
                                 snapshot.case_id, snapshot.rep, std::string_view(snapshot.algo), snapshot.ncv, snapshot.block_size, snapshot.max_retain_blocks,
                                 std::string_view(snapshot.ritz), bool_text(snapshot.use_refined_rayleigh_ritz), snapshot.abstol,
                                 std::string_view(snapshot.stop_reason), snapshot.eigenvalue, snapshot.rnorm_abs, snapshot.rnorm_rel, snapshot.outer_iterations,
                                 snapshot.matvecs, snapshot.outer_matvecs, snapshot.time, mem_size(snapshot.vmrss_mib), mem_size(snapshot.vmhwm_mib));
                } else {
                    std::println("{:<5} {:<5} {:<8} {:>5} {:>5} {:<4} {:<7} {:>10.2e} {:<24} {:>18.10e} {:>12.4e} {:>12.4e} {:>8} {:>8} {:>8} {:>12.6f} {:>12} "
                                 "{:>12}",
                                 snapshot.case_id, snapshot.rep, std::string_view(snapshot.algo), snapshot.ncv, snapshot.block_size,
                                 std::string_view(snapshot.ritz), bool_text(snapshot.use_refined_rayleigh_ritz), snapshot.abstol,
                                 std::string_view(snapshot.stop_reason), snapshot.eigenvalue, snapshot.rnorm_abs, snapshot.rnorm_rel, snapshot.outer_iterations,
                                 snapshot.matvecs, snapshot.outer_matvecs, snapshot.time, mem_size(snapshot.vmrss_mib), mem_size(snapshot.vmhwm_mib));
                }
            },
            result);
    }
}

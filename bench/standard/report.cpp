#include "report.h"

#include "memory.h"

#include <Eigen/Core>
#include <format>
#include <print>
#include <string>
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

    std::string_view bool_text(bool value) {
        return value ? "true" : "false";
    }

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

    std::string limit_text(Eigen::Index value) {
        return value < 0 ? "unlimited" : std::format("{}", value);
    }

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

    void print_solver_config(const Options &opts, const grit::gdplusk_config<Scalar> &cfg) {
        std::println("solver:");
        std::println("  problem: standard | outer iteration: gdplusk | ritz: {}", grit::enum2sv(cfg.ritz));
        std::println("  residual correction: {}", residual_correction_name(opts.residual_correction));
        std::println("  refined Rayleigh-Ritz: {} | relative rnorm tolerance: {} | adaptive inner tolerance: {}",
                     bool_text(opts.refined_rayleigh_ritz), bool_text(opts.use_relative_rnorm_tolerance), bool_text(opts.use_adaptive_inner_tolerance));
        std::println("  nev: {} | ncv: {} | block size: {} | max-iters: {} | max-matvecs: {}", cfg.nev, cfg.ncv, cfg.block_size, limit_text(cfg.max_iters),
                     limit_text(cfg.max_matvecs));
        std::println("  tol: {:.4e} | tol rnorm relative: {:.4e} | inner tol: {:.4e} | inner max iters: {}", cfg.tol, cfg.tol_rnorm_relative, cfg.inner_tol,
                     cfg.inner_max_iters);
        std::println("  initial guess: {} | seed: {} | Eigen threads: {} | log level: {}",
                     opts.initial_guess.empty() ? "random(seed + rep - 1)" : opts.initial_guess, opts.seed, Eigen::nbThreads(), log_level_name(cfg.log_level));
        std::println("  history: extra ritz {} | ritz residual {} | basis blocks {} | retain blocks {}", cfg.max_extra_ritz_history,
                     cfg.max_ritz_residual_history, cfg.max_basis_blocks, cfg.max_retain_blocks);
        if(opts.residual_correction == ResidualCorrection::AUTO) {
            std::println("  auto correction: cheap min dwell iters {} | eigval saturation {:.3e} | rnorm saturation {:.3e} | jd start rnorm {:.3e} | cheap probe interval {} | "
                         "cheap probe factor {:.3e}",
                         cfg.auto_min_dwell_iters, cfg.auto_sat_eigval_threshold, cfg.auto_sat_rnorm_threshold, cfg.auto_jd_start_rnorm_threshold,
                         cfg.auto_cheap_probe_interval, cfg.auto_cheap_probe_factor);
        }
        std::println("  matvec: Eigen row-major sparse * dense");
    }

    void print_sweep_config(const CliOptions &opts, std::size_t cases) {
        std::println("sweep:");
        std::println("  cases: {} | reps: {} | seed: {} | seed per repetition: seed + rep - 1", cases, opts.reps, opts.seed);
        std::println("  limits: max-iters {} | max-matvecs {} | tol rnorm relative {:.4e} | relative rnorm tolerance: {} | eigval saturation {:.4e} | rnorm saturation {:.4e}",
                     limit_text(opts.max_iters), limit_text(opts.max_matvecs), opts.tol_rnorm_relative, bool_text(opts.use_relative_rnorm_tolerance),
                     opts.sat_eigval_threshold, opts.sat_rnorm_threshold);
        std::println("  sweep axes: ncv {} | block-size {} | max basis blocks {} | ritz {} | residual correction {}", list_text(opts.ncv),
                     list_text(opts.block_size), opts.max_basis_blocks, opts.ritz, opts.residual_correction);
        std::println("  sweep axes: tol {} | inner tol {} | inner max iters {} | refined {} | adaptive {}", list_text(opts.tol), list_text(opts.inner_tol),
                     list_text(opts.inner_max_iters), bool_list_text(opts.refined_rayleigh_ritz), bool_list_text(opts.use_adaptive_inner_tolerance));
        if(opts.residual_correction.find("auto") != std::string::npos || opts.residual_correction.find("AUTO") != std::string::npos) {
            std::println("  auto correction: cheap min dwell iters {} | eigval saturation {:.3e} | rnorm saturation {:.3e} | jd start rnorm {:.3e} | cheap probe interval {} | "
                         "cheap probe factor {:.3e}",
                         opts.auto_min_dwell_iters, opts.auto_sat_eigval_threshold, opts.auto_sat_rnorm_threshold, opts.auto_jd_start_rnorm_threshold,
                         opts.auto_cheap_probe_interval, opts.auto_cheap_probe_factor);
        }
        std::println("  initial guess: {}", opts.initial_guess.empty() ? "random" : opts.initial_guess);
        if(!opts.save_eigvec.empty()) std::println("  save eigvec: {}", opts.save_eigvec);
        if(!opts.save_results.empty()) std::println("  save results: {}", opts.save_results);
        std::println("  Eigen threads: {} | log level: {}", Eigen::nbThreads(), log_level_name(opts.log_level));
        std::println("  matvec: Eigen row-major sparse * dense");
    }

    void print_result_header() {
        std::println("{:<5} {:<5} {:>5} {:>5} {:<4} {:<17} {:<7} {:<8} {:>10} {:>10} {:>7} {:<24} {:>18} {:>12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>12} {:>12} {:>12} {:>12}",
                     "case", "rep", "ncv", "blk", "ritz", "correction", "refined", "adaptive", "tol", "inner_tol", "inner", "stop", "eigval", "rnorm",
                     "iter", "matvec", "outer", "inner", "jdops", "jd_switch", "time[s]", "VmRSS", "VmHWM", "VmPeak");
    }

    void print_result_row(const SolveResult &result) {
        const auto &opts      = result.options;
        const auto jd_switch = result.first_cheap_to_jd_iter < 0 ? std::string{"n/a"} : std::format("{}", result.first_cheap_to_jd_iter);
        std::println("{:<5} {:<5} {:>5} {:>5} {:<4} {:<17} {:<7} {:<8} {:>10.2e} {:>10.2e} {:>7} {:<24} {:>18.10e} {:>12.4e} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>12.6f} {:>12} {:>12} {:>12}",
                     opts.case_id, result.rep, opts.ncv, opts.block_size, grit::enum2sv(opts.ritz), residual_correction_name(opts.residual_correction),
                     bool_text(opts.refined_rayleigh_ritz), bool_text(opts.use_adaptive_inner_tolerance), opts.tol, result.inner_tol, result.inner_iters,
                     result.stop_reason, result.eigenvalue, result.residual, result.iterations, result.matvecs, result.outer_matvecs, result.inner_matvecs,
                     result.jdops_inner, jd_switch, result.seconds, mem_size(result.vmrss_mib), mem_size(result.vmhwm_mib), mem_size(result.vmpeak_mib));
    }
}

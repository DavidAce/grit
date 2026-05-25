#include "report.h"

#include "memory.h"

#include <Eigen/Core>
#include <format>
#include <print>

namespace bench_standard {
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
                     bool_text(opts.refined_rayleigh_ritz), bool_text(opts.relative_rnorm_tolerance), bool_text(opts.adaptive_inner_tolerance));
        std::println("  nev: {} | ncv: {} | block size: {} | max-iters: {} | max-matvecs: {}", cfg.nev, cfg.ncv, cfg.b, limit_text(cfg.max_iters),
                     limit_text(cfg.max_matvecs));
        std::println("  abstol: {:.4e} | reltol: {:.4e} | inner tol: {:.4e} | inner iters: {}", cfg.abstol, cfg.reltol, cfg.inner_tol,
                     cfg.inner_iters);
        std::println("  initial guess: {} | seed: {} | Eigen threads: {} | log level: {}",
                     opts.initial_guess.empty() ? "random(seed + rep - 1)" : opts.initial_guess, opts.seed, Eigen::nbThreads(), log_level_name(cfg.logLevel));
        std::println("  history: extra ritz {} | ritz residual {} | basis blocks {} | retain blocks {}", cfg.maxExtraRitzHistory,
                     cfg.maxRitzResidualHistory, cfg.maxBasisBlocks, cfg.maxRetainBlocks);
        std::println("  matvec: Eigen row-major sparse * dense");
    }

    void print_sweep_config(const CliOptions &opts, std::size_t cases) {
        std::println("sweep:");
        std::println("  cases: {} | reps: {} | seed: {} | seed per repetition: seed + rep - 1", cases, opts.reps, opts.seed);
        std::println("  limits: max-iters {} | max-matvecs {} | reltol {:.4e} | relative rnorm tolerance: {} | eval stall tol {:.4e} | rnorm stall tol {:.4e}",
                     limit_text(opts.max_iters), limit_text(opts.max_matvecs), opts.reltol, bool_text(opts.relative_rnorm_tolerance),
                     opts.tol_stall_evals, opts.tol_stall_rnorm);
        std::println("  sweep axes: ncv {} | block-size {} | max basis blocks {} | ritz {} | residual correction {}", opts.ncv, opts.block_size,
                     opts.max_basis_blocks, opts.ritz, opts.residual_correction);
        std::println("  sweep axes: tol {} | inner tol {} | inner iters {} | refined {} | adaptive {}", opts.tol, opts.inner_tol, opts.inner_iters,
                     opts.refined_rayleigh_ritz, opts.adaptive_inner_tolerance);
        std::println("  initial guess: {}", opts.initial_guess.empty() ? "random" : opts.initial_guess);
        if(!opts.save_eigvec.empty()) std::println("  save eigvec: {}", opts.save_eigvec);
        std::println("  Eigen threads: {} | log level: {}", Eigen::nbThreads(), log_level_name(opts.log_level));
        std::println("  matvec: Eigen row-major sparse * dense");
    }

    void print_result_header() {
        std::println("{:<5} {:<5} {:>5} {:>5} {:<4} {:<17} {:<7} {:<8} {:>10} {:>10} {:>7} {:<24} {:>18} {:>12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>12} {:>12} {:>12} {:>12}",
                     "case", "rep", "ncv", "blk", "ritz", "correction", "refined", "adaptive", "tol", "inner_tol", "inner", "stop", "eigval", "rnorm",
                     "iter", "matvec", "outer", "inner", "jdops", "time[s]", "VmRSS", "VmHWM", "VmPeak");
    }

    void print_result_row(const SolveResult &result) {
        std::println("{:<5} {:<5} {:>5} {:>5} {:<4} {:<17} {:<7} {:<8} {:>10.2e} {:>10.2e} {:>7} {:<24} {:>18.10e} {:>12.4e} {:>8} {:>8} {:>8} {:>8} {:>8} {:>12.6f} {:>12} {:>12} {:>12}",
                     result.case_id, result.rep, result.ncv, result.block_size, grit::enum2sv(result.ritz), residual_correction_name(result.residual_correction),
                     bool_text(result.refined_rayleigh_ritz), bool_text(result.adaptive_inner_tolerance), result.tol, result.inner_tol, result.inner_iters,
                     result.stop_reason, result.eigenvalue, result.residual, result.iterations, result.matvecs, result.outer_matvecs, result.inner_matvecs,
                     result.jdops_inner, result.seconds, mem_size(result.vmrss_mib), mem_size(result.vmhwm_mib), mem_size(result.vmpeak_mib));
    }
}

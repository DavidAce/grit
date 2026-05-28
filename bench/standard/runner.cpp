#include "runner.h"
#include "hdf5_io.h"
#include "memory.h"
#include <chrono>
#include <format>
#include <random>
#include <stdexcept>

namespace bench_standard {
    namespace {
        DenseMatrix random_initial_guess(Eigen::Index rows, Eigen::Index cols, unsigned int seed) {
            DenseMatrix                            guess(rows, cols);
            std::mt19937                           rng(seed);
            std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);
            for(Eigen::Index col = 0; col < cols; ++col)
                for(Eigen::Index row = 0; row < rows; ++row) guess(row, col) = dist(rng);
            return guess;
        }

        DenseMatrix initial_guess(const Options &opts, Eigen::Index rows, int rep) {
            if(opts.initial_guess.empty()) return random_initial_guess(rows, opts.block_size, opts.seed + static_cast<unsigned int>(rep - 1));

            auto guess = load_initial_guess_hdf5(opts.initial_guess);
            if(guess.rows() != rows) { throw std::runtime_error(std::format("Initial guess rows ({}) do not match matrix rows ({})", guess.rows(), rows)); }
            if(guess.cols() < opts.nev) {
                throw std::runtime_error(std::format("Initial guess columns ({}) must be at least nev ({})", guess.cols(), opts.nev));
            }
            return guess;
        }

        void apply_solver_options(Solver &solver, const Options &opts) {
            solver.residual_correction_type     = opts.residual_correction;
            solver.use_refined_rayleigh_ritz    = opts.refined_rayleigh_ritz;
            solver.use_relative_rnorm_tolerance = opts.use_relative_rnorm_tolerance;
            solver.use_adaptive_inner_tolerance = opts.use_adaptive_inner_tolerance;
        }
    }

    grit::gdplusk_config<Scalar> make_solver_config(const Options &opts) {
        grit::gdplusk_config<Scalar> cfg;
        cfg.nev                      = opts.nev;
        cfg.ncv                      = opts.ncv;
        cfg.block_size                        = opts.block_size;
        cfg.max_basis_blocks           = opts.max_basis_blocks;
        cfg.ritz                     = opts.ritz;
        cfg.max_iters                = opts.max_iters;
        cfg.max_matvecs              = opts.max_matvecs;
        cfg.inner_max_iters          = opts.inner_max_iters;
        cfg.tol                   = opts.tol;
        cfg.tol_rnorm_relative                   = opts.use_relative_rnorm_tolerance ? (opts.tol_rnorm_relative > 0.0 ? opts.tol_rnorm_relative : opts.tol) : 0.0;
        cfg.sat_eigval_threshold          = opts.sat_eigval_threshold;
        cfg.sat_rnorm_threshold          = opts.sat_rnorm_threshold;
        cfg.inner_tol                = opts.inner_tol;
        cfg.auto_min_dwell_iters     = opts.auto_min_dwell_iters;
        cfg.auto_sat_eigval_threshold      = opts.auto_sat_eigval_threshold;
        cfg.auto_sat_rnorm_threshold     = opts.auto_sat_rnorm_threshold;
        cfg.auto_jd_start_rnorm_threshold = opts.auto_jd_start_rnorm_threshold;
        cfg.auto_cheap_probe_interval = opts.auto_cheap_probe_interval;
        cfg.auto_cheap_probe_factor   = opts.auto_cheap_probe_factor;
        cfg.log_level                 = opts.log_level;
        return cfg;
    }

    SolveResult solve_once(const SparseMatrix &matrix, Options opts, int rep) {
        auto Aop = grit::matvec<Scalar>(matrix.rows(), [&](const DenseMatrixRef &X) -> DenseMatrix { return matrix * X; });

        auto guess = initial_guess(opts, matrix.rows(), rep);

        grit::standard::problem<Scalar> problem(Aop);
        auto                            cfg = make_solver_config(opts);
        cfg.set_initial_guess(guess);

        const auto time_start = std::chrono::steady_clock::now();
        Solver     solver(problem, cfg);
        apply_solver_options(solver, opts);
        solver.run();
        const auto time_stop = std::chrono::steady_clock::now();

        const auto  result = solver.result();
        SolveResult solve_result{
            .options                  = opts,
            .rep                      = rep,
            .rep_seed                 = opts.seed + static_cast<unsigned int>(rep - 1),
            .inner_iters              = static_cast<int>(solver.inner_max_iters),
            .inner_tol                = solver.inner_tol,
            .stop_reason              = grit::enum2s(result.stopReason()),
            .eigenvalue               = result.eigVal().size() > 0 ? result.eigVal()(0) : 0.0,
            .residual                 = result.rNorms().size() > 0 ? result.rNorms()(0) : 0.0,
            .iterations               = result.iter(),
            .matvecs                  = result.num_matvecs_total(),
            .outer_matvecs            = result.num_matvecs(),
            .inner_matvecs            = result.num_matvecs_inner(),
            .jdops_inner              = result.num_jdops_inner(),
            .first_cheap_to_jd_iter   = result.cheap_to_jd_switch_iters().empty() ? Eigen::Index{-1} : result.cheap_to_jd_switch_iters().front(),
            .cheap_to_jd_switch_iters = result.cheap_to_jd_switch_iters(),
            .jd_to_cheap_switch_iters = result.jd_to_cheap_switch_iters(),
            .seconds                  = std::chrono::duration<double>(time_stop - time_start).count(),
            .vmrss_mib                = mem_usage_in_mib("VmRSS"),
            .vmhwm_mib                = mem_usage_in_mib("VmHWM"),
            .vmpeak_mib               = mem_usage_in_mib("VmPeak"),
        };
        if(result.eigVecs().cols() >= opts.nev) solve_result.eigvecs = result.eigVecs().leftCols(opts.nev);
        return solve_result;
    }
}

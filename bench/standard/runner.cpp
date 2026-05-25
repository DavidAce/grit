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
            if(guess.rows() != rows) {
                throw std::runtime_error(std::format("Initial guess rows ({}) do not match matrix rows ({})", guess.rows(), rows));
            }
            if(guess.cols() < opts.nev) {
                throw std::runtime_error(std::format("Initial guess columns ({}) must be at least nev ({})", guess.cols(), opts.nev));
            }
            return guess;
        }

        void apply_solver_options(Solver &solver, const Options &opts) {
            solver.residual_correction_type      = opts.residual_correction;
            solver.use_refined_rayleigh_ritz     = opts.refined_rayleigh_ritz;
            solver.use_relative_rnorm_tolerance  = opts.relative_rnorm_tolerance;
            solver.use_adaptive_inner_tolerance  = opts.adaptive_inner_tolerance;
        }
    }

    grit::gdplusk_config<Scalar> make_solver_config(const Options &opts) {
        grit::gdplusk_config<Scalar> cfg;
        cfg.nev         = opts.nev;
        cfg.ncv         = opts.ncv;
        cfg.b           = opts.block_size;
        cfg.maxBasisBlocks = opts.max_basis_blocks;
        cfg.ritz        = opts.ritz;
        cfg.max_iters   = opts.max_iters;
        cfg.max_matvecs = opts.max_matvecs;
        cfg.inner_iters = opts.inner_iters;
        cfg.abstol      = opts.tol;
        cfg.reltol      = opts.relative_rnorm_tolerance ? (opts.reltol > 0.0 ? opts.reltol : opts.tol) : 0.0;
        cfg.tol_stall_evals  = opts.tol_stall_evals;
        cfg.tol_stall_rnorm = opts.tol_stall_rnorm;
        cfg.inner_tol   = opts.inner_tol;
        cfg.logLevel    = opts.log_level;
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

        const auto result = solver.result();
        SolveResult solve_result{
            .case_id                  = opts.case_id,
            .rep                      = rep,
            .ncv                      = opts.ncv,
            .block_size               = opts.block_size,
            .inner_iters              = static_cast<int>(solver.inner_iters),
            .tol                      = opts.tol,
            .inner_tol                = solver.inner_tol,
            .ritz                     = opts.ritz,
            .residual_correction      = opts.residual_correction,
            .refined_rayleigh_ritz    = opts.refined_rayleigh_ritz,
            .adaptive_inner_tolerance = opts.adaptive_inner_tolerance,
            .stop_reason              = grit::enum2s(result.stopReason()),
            .eigenvalue               = result.eigVal().size() > 0 ? result.eigVal()(0) : 0.0,
            .residual                 = result.rNorms().size() > 0 ? result.rNorms()(0) : 0.0,
            .iterations               = result.iter(),
            .matvecs                  = result.num_matvecs_total(),
            .outer_matvecs            = result.num_matvecs(),
            .inner_matvecs            = result.num_matvecs_inner(),
            .jdops_inner              = result.num_jdops_inner(),
            .seconds                  = std::chrono::duration<double>(time_stop - time_start).count(),
            .vmrss_mib                = mem_usage_in_mib("VmRSS"),
            .vmhwm_mib                = mem_usage_in_mib("VmHWM"),
            .vmpeak_mib               = mem_usage_in_mib("VmPeak"),
        };
        if(result.eigVecs().cols() >= opts.nev) solve_result.eigvecs = result.eigVecs().leftCols(opts.nev);
        return solve_result;
    }
}

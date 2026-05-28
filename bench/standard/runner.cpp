#include "runner.h"
#include "hdf5_io.h"
#include "memory.h"
#include "report.h"
#include <chrono>
#include <cstdint>
#include <format>
#include <limits>
#include <random>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

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
            solver.use_refined_rayleigh_ritz    = opts.use_refined_rayleigh_ritz;
            solver.use_relative_rnorm_tolerance = opts.use_relative_rnorm_tolerance;
            solver.use_adaptive_inner_tolerance = opts.use_adaptive_inner_tolerance;
        }

        template<std::size_t N>
        h5pp::fstr_t<N> fixed_string(std::string_view text, std::string_view field_name) {
            if(text.size() >= N) throw std::runtime_error(std::format("HDF5 field '{}' is too long: {} >= {}", field_name, text.size(), N));
            return h5pp::fstr_t<N>{text};
        }

        std::vector<int64_t> to_i64_vector(const std::vector<Eigen::Index> &values) {
            std::vector<int64_t> result;
            result.reserve(values.size());
            for(auto value : values) result.push_back(static_cast<int64_t>(value));
            return result;
        }

        double relative_rnorm(grit::result_view<Scalar> result) {
            if(result.rNorms().size() == 0) return 0.0;
            auto scale = std::max(result.op_norm_estimate(), std::numeric_limits<double>::min());
            if(result.eigVecs().cols() > 0) scale = std::max(scale * result.eigVecs().col(0).norm(), std::numeric_limits<double>::min());
            return result.rNorms()(0) / scale;
        }

        SolverSnapshot make_solver_snapshot(const Options &opts, int rep, unsigned int rep_seed, const Solver &solver, grit::result_view<Scalar> result) {
            SolverSnapshot snapshot;
            snapshot.case_id                       = static_cast<int32_t>(opts.case_id);
            snapshot.rep                           = static_cast<int32_t>(rep);
            snapshot.seed                          = opts.seed;
            snapshot.rep_seed                      = rep_seed;
            snapshot.matrix_path                   = fixed_string<1024>(opts.matrix_path, "matrix_path");
            snapshot.initial_guess                 = fixed_string<1024>(opts.initial_guess, "initial_guess");
            snapshot.nev                           = static_cast<int32_t>(opts.nev);
            snapshot.ncv                           = static_cast<int32_t>(opts.ncv);
            snapshot.block_size                    = static_cast<int32_t>(opts.block_size);
            snapshot.max_basis_blocks              = static_cast<int32_t>(opts.max_basis_blocks);
            snapshot.max_iters                     = static_cast<int32_t>(opts.max_iters);
            snapshot.max_matvecs                   = static_cast<int32_t>(opts.max_matvecs);
            snapshot.inner_max_iters               = static_cast<int32_t>(solver.inner_max_iters);
            snapshot.tol                           = opts.tol;
            snapshot.tol_rnorm_relative            = opts.tol_rnorm_relative;
            snapshot.sat_eigval_threshold          = opts.sat_eigval_threshold;
            snapshot.sat_rnorm_threshold           = opts.sat_rnorm_threshold;
            snapshot.inner_tol                     = solver.inner_tol;
            snapshot.auto_min_dwell_iters          = static_cast<int32_t>(opts.auto_min_dwell_iters);
            snapshot.auto_sat_eigval_threshold     = opts.auto_sat_eigval_threshold;
            snapshot.auto_sat_rnorm_threshold      = opts.auto_sat_rnorm_threshold;
            snapshot.auto_jd_start_rnorm_threshold = opts.auto_jd_start_rnorm_threshold;
            snapshot.auto_cheap_probe_interval     = static_cast<int32_t>(opts.auto_cheap_probe_interval);
            snapshot.auto_cheap_probe_factor       = opts.auto_cheap_probe_factor;
            snapshot.ritz                          = fixed_string<16>(std::string(grit::enum2sv(opts.ritz)), "ritz");
            snapshot.residual_correction           = fixed_string<32>(residual_correction_name(opts.residual_correction), "residual_correction");
            snapshot.use_refined_rayleigh_ritz     = static_cast<uint8_t>(opts.use_refined_rayleigh_ritz);
            snapshot.use_relative_rnorm_tolerance  = static_cast<uint8_t>(opts.use_relative_rnorm_tolerance);
            snapshot.use_adaptive_inner_tolerance  = static_cast<uint8_t>(opts.use_adaptive_inner_tolerance);
            snapshot.inner_tol_last                = result.inner_tol_last();
            snapshot.inner_error_last              = result.inner_error_last();
            snapshot.stop_reason                   = fixed_string<64>(grit::enum2s(result.stopReason()), "stop_reason");
            snapshot.eigenvalue                    = result.eigVal().size() > 0 ? result.eigVal()(0) : 0.0;
            snapshot.rnorm                         = result.rNorms().size() > 0 ? result.rNorms()(0) : 0.0;
            snapshot.rrnorm                        = relative_rnorm(result);
            snapshot.iterations                    = static_cast<int64_t>(result.iter());
            snapshot.matvecs                       = static_cast<int64_t>(result.num_matvecs_total());
            snapshot.outer_matvecs                 = static_cast<int64_t>(result.num_matvecs());
            snapshot.inner_matvecs                 = static_cast<int64_t>(result.num_matvecs_inner());
            snapshot.inner_iterations              = static_cast<int64_t>(result.num_iters_inner());
            snapshot.jdops_inner                   = static_cast<int64_t>(result.num_jdops_inner());
            snapshot.precond                       = static_cast<int64_t>(result.num_precond());
            snapshot.precond_inner                 = static_cast<int64_t>(result.num_precond_inner());
            snapshot.precond_total                 = static_cast<int64_t>(result.num_precond_total());
            snapshot.first_cheap_to_jd_iter =
                result.cheap_to_jd_switch_iters().empty() ? int64_t{-1} : static_cast<int64_t>(result.cheap_to_jd_switch_iters().front());
            snapshot.auto_dwell                 = static_cast<int64_t>(result.auto_dwell());
            snapshot.auto_jd_steps_since_probe  = static_cast<int64_t>(result.auto_jd_steps_since_probe());
            snapshot.saturation_count_eigval    = static_cast<int64_t>(result.saturation_count_eigval());
            snapshot.saturation_count_rnorm     = static_cast<int64_t>(result.saturation_count_rnorm());
            snapshot.saturation_count_max       = static_cast<int64_t>(result.saturation_count_max());
            snapshot.op_norm_estimate           = result.op_norm_estimate();
            snapshot.condition                  = result.condition();
            snapshot.sensitivity                = result.sensitivity();
            snapshot.gap                        = result.gap();
            snapshot.rnorm_below_tol            = static_cast<uint8_t>(result.rnorm_below_tol());
            snapshot.rnorm_below_gap            = static_cast<uint8_t>(result.rnorm_below_gap());
            snapshot.residual_correction_active = fixed_string<32>(result.residual_correction_active_name(), "residual_correction_active");
            snapshot.residual_correction_step   = fixed_string<32>(result.residual_correction_step_name(), "residual_correction_step");
            snapshot.num_cheap_to_jd_switches   = static_cast<int64_t>(result.cheap_to_jd_switch_iters().size());
            snapshot.num_jd_to_cheap_switches   = static_cast<int64_t>(result.jd_to_cheap_switch_iters().size());
            snapshot.cheap_to_jd_switch_iters   = to_i64_vector(result.cheap_to_jd_switch_iters());
            snapshot.jd_to_cheap_switch_iters   = to_i64_vector(result.jd_to_cheap_switch_iters());
            snapshot.seconds                    = result.seconds();
            return snapshot;
        }
    }

    grit::gdplusk_config<Scalar> make_solver_config(const Options &opts) {
        grit::gdplusk_config<Scalar> cfg;
        cfg.nev                           = opts.nev;
        cfg.ncv                           = opts.ncv;
        cfg.block_size                    = opts.block_size;
        cfg.max_basis_blocks              = opts.max_basis_blocks;
        cfg.ritz                          = opts.ritz;
        cfg.max_iters                     = opts.max_iters;
        cfg.max_matvecs                   = opts.max_matvecs;
        cfg.inner_max_iters               = opts.inner_max_iters;
        cfg.tol                           = opts.tol;
        cfg.tol_rnorm_relative            = opts.use_relative_rnorm_tolerance ? (opts.tol_rnorm_relative > 0.0 ? opts.tol_rnorm_relative : opts.tol) : 0.0;
        cfg.sat_eigval_threshold          = opts.sat_eigval_threshold;
        cfg.sat_rnorm_threshold           = opts.sat_rnorm_threshold;
        cfg.inner_tol                     = opts.inner_tol;
        cfg.auto_min_dwell_iters          = opts.auto_min_dwell_iters;
        cfg.auto_sat_eigval_threshold     = opts.auto_sat_eigval_threshold;
        cfg.auto_sat_rnorm_threshold      = opts.auto_sat_rnorm_threshold;
        cfg.auto_jd_start_rnorm_threshold = opts.auto_jd_start_rnorm_threshold;
        cfg.auto_cheap_probe_interval     = opts.auto_cheap_probe_interval;
        cfg.auto_cheap_probe_factor       = opts.auto_cheap_probe_factor;
        cfg.log_level                     = opts.log_level;
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
        const auto                  rep_seed = opts.seed + static_cast<unsigned int>(rep - 1);
        std::vector<SolverSnapshot> snapshots;
        solver.status_callback = [&](grit::result_view<Scalar> status) { snapshots.push_back(make_solver_snapshot(opts, rep, rep_seed, solver, status)); };
        solver.run();
        const auto time_stop = std::chrono::steady_clock::now();

        const auto result         = solver.result();
        auto       final_snapshot = make_solver_snapshot(opts, rep, rep_seed, solver, result);
        final_snapshot.seconds    = std::chrono::duration<double>(time_stop - time_start).count();
        final_snapshot.vmrss_mib  = mem_usage_in_mib("VmRSS");
        final_snapshot.vmhwm_mib  = mem_usage_in_mib("VmHWM");
        final_snapshot.vmpeak_mib = mem_usage_in_mib("VmPeak");
        if(snapshots.empty())
            snapshots.push_back(final_snapshot);
        else
            snapshots.back() = final_snapshot;

        SolveResult solve_result{
            .final     = final_snapshot,
            .snapshots = std::move(snapshots),
        };
        if(result.eigVecs().cols() >= opts.nev) solve_result.eigvecs = result.eigVecs().leftCols(opts.nev);
        return solve_result;
    }
}

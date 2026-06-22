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
#include <type_traits>
#include <utility>
#include <vector>

namespace bench_generalized {
    namespace {
        using GdSolver      = grit::generalized::gdplusk<Scalar>;
        using LanczosSolver = grit::generalized::lanczos<Scalar>;
        using LobpcgSolver  = grit::generalized::lobpcg<Scalar>;

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

            auto guess = load_initial_guess_hdf5(opts.initial_guess, opts.algo);
            if(guess.rows() != rows) { throw std::runtime_error(std::format("Initial guess rows ({}) do not match matrix rows ({})", guess.rows(), rows)); }
            if(guess.cols() < opts.nev) {
                throw std::runtime_error(std::format("Initial guess columns ({}) must be at least nev ({})", guess.cols(), opts.nev));
            }
            return guess;
        }

        template<std::size_t N>
        h5pp::fstr_t<N> fixed_string(std::string_view text, std::string_view field_name) {
            if(text.size() >= N) throw std::runtime_error(std::format("HDF5 field '{}' is too long: {} >= {}", field_name, text.size(), N));
            return h5pp::fstr_t<N>{text};
        }

        std::vector<int64_t> to_i64_vector(const std::vector<Eigen::Index> &values) {
            std::vector<int64_t> converted;
            converted.reserve(values.size());
            for(auto value : values) converted.push_back(static_cast<int64_t>(value));
            return converted;
        }

        double rnorm_rel(grit::ResultView<Scalar> view) {
            if(view.rNormsAbs().size() == 0) return 0.0;
            auto scale = std::max(view.op_norm_estimate(), std::numeric_limits<double>::min());
            if(view.eigVecs().cols() > 0) scale = std::max(scale * view.eigVecs().col(0).norm(), std::numeric_limits<double>::min());
            return view.rNormsAbs()(0) / scale;
        }

        template<typename Snapshot>
        void fill_common_snapshot(Snapshot &snapshot, const Options &opts, int rep, unsigned int rep_seed, grit::ResultView<Scalar> view) {
            snapshot.case_id                       = static_cast<int32_t>(opts.case_id);
            snapshot.rep                           = static_cast<int32_t>(rep);
            snapshot.seed                          = opts.seed;
            snapshot.rep_seed                      = rep_seed;
            snapshot.matrix_a_path                 = fixed_string<1024>(opts.matrix_a_path, "matrix_a_path");
            snapshot.matrix_b_path                 = fixed_string<1024>(opts.matrix_b_path, "matrix_b_path");
            snapshot.initial_guess                 = fixed_string<1024>(opts.initial_guess, "initial_guess");
            snapshot.algo                          = fixed_string<32>(algo_name(opts.algo), "algo");
            snapshot.nev                           = static_cast<int32_t>(opts.nev);
            snapshot.ncv                           = static_cast<int32_t>(opts.ncv);
            snapshot.block_size                    = static_cast<int32_t>(opts.block_size);
            snapshot.max_iters                     = static_cast<int32_t>(opts.max_iters);
            snapshot.max_matvecs                   = static_cast<int32_t>(opts.max_matvecs);
            snapshot.abstol                           = opts.abstol;
            snapshot.reltol            = opts.reltol;
            snapshot.sat_eigval_threshold          = opts.sat_eigval_threshold;
            snapshot.sat_rnorm_threshold           = opts.sat_rnorm_threshold;
            snapshot.ritz                          = fixed_string<16>(std::string(grit::enum2sv(opts.ritz)), "ritz");
            snapshot.use_refined_rayleigh_ritz     = static_cast<uint8_t>(opts.use_refined_rayleigh_ritz);
            snapshot.use_b_inner_product           = static_cast<uint8_t>(opts.use_b_inner_product);
            snapshot.use_rescaled_rnorm_tolerance  = static_cast<uint8_t>(opts.use_rescaled_rnorm_tolerance);
            snapshot.stop_reason                   = fixed_string<64>(grit::enum2s(view.stopReason()), "stop_reason");
            snapshot.eigenvalue                    = view.eigVal().size() > 0 ? view.eigVal()(0) : 0.0;
            snapshot.rnorm_abs                         = view.rNormsAbs().size() > 0 ? view.rNormsAbs()(0) : 0.0;
            snapshot.rnorm_rel                        = rnorm_rel(view);
            snapshot.outer_iterations                    = static_cast<int64_t>(view.outer_iter());
            snapshot.matvecs                       = static_cast<int64_t>(view.num_matvecs_total());
            snapshot.outer_matvecs                 = static_cast<int64_t>(view.num_matvecs());
            snapshot.inner_matvecs                 = static_cast<int64_t>(view.num_matvecs_inner());
            snapshot.inner_iterations              = static_cast<int64_t>(view.num_inner_iters());
            snapshot.jdops_inner                   = static_cast<int64_t>(view.num_jdops_inner());
            snapshot.precond                       = static_cast<int64_t>(view.num_precond());
            snapshot.precond_inner                 = static_cast<int64_t>(view.num_precond_inner());
            snapshot.precond_total                 = static_cast<int64_t>(view.num_precond_total());
            snapshot.saturation_count_eigval       = static_cast<int64_t>(view.saturation_count_eigval());
            snapshot.saturation_count_rnorm        = static_cast<int64_t>(view.saturation_count_rnorm());
            snapshot.saturation_count_max          = static_cast<int64_t>(view.saturation_count_max());
            snapshot.op_norm_estimate              = view.op_norm_estimate();
            snapshot.condition                     = view.condition();
            snapshot.sensitivity                   = view.sensitivity();
            snapshot.gap                           = view.gap();
            snapshot.residual_converged               = static_cast<uint8_t>(view.residual_converged());
            snapshot.residual_below_gap               = static_cast<uint8_t>(view.residual_below_gap());
            snapshot.time                       = view.time();
        }

        GdpluskSnapshot make_solver_snapshot(const Options &opts, int rep, unsigned int rep_seed, const GdSolver &solver, grit::ResultView<Scalar> view) {
            GdpluskSnapshot snapshot;
            fill_common_snapshot(snapshot, opts, rep, rep_seed, view);
            snapshot.max_basis_blocks              = static_cast<int32_t>(opts.ncv / opts.block_size);
            snapshot.inner_max_iters               = static_cast<int32_t>(solver.config.inner_max_iters);
            snapshot.inner_tol                     = solver.config.inner_tol;
            snapshot.auto_min_dwell_iters          = static_cast<int32_t>(opts.auto_min_dwell_iters);
            snapshot.auto_sat_eigval_threshold     = opts.auto_sat_eigval_threshold;
            snapshot.auto_sat_rnorm_threshold      = opts.auto_sat_rnorm_threshold;
            snapshot.auto_jd_start_rnorm_threshold = opts.auto_jd_start_rnorm_threshold;
            snapshot.auto_cheap_probe_interval     = static_cast<int32_t>(opts.auto_cheap_probe_interval);
            snapshot.auto_cheap_probe_factor       = opts.auto_cheap_probe_factor;
            snapshot.residual_correction           = fixed_string<32>(residual_correction_name(opts.residual_correction), "residual_correction");
            snapshot.use_jd_b_only                 = static_cast<uint8_t>(opts.use_jd_b_only);
            snapshot.use_adaptive_inner_tolerance  = static_cast<uint8_t>(opts.use_adaptive_inner_tolerance);
            snapshot.inner_tol_last                = view.inner_tol_last();
            snapshot.inner_error_last              = view.inner_error_last();
            snapshot.first_cheap_to_jd_outer_iter =
                view.cheap_to_jd_switch_outer_iters().empty() ? int64_t{-1} : static_cast<int64_t>(view.cheap_to_jd_switch_outer_iters().front());
            snapshot.auto_dwell                 = static_cast<int64_t>(view.auto_dwell());
            snapshot.auto_jd_outer_iters_since_probe  = static_cast<int64_t>(view.auto_jd_outer_iters_since_probe());
            snapshot.residual_correction_active = fixed_string<32>(view.residual_correction_active_name(), "residual_correction_active");
            snapshot.residual_correction_iteration   = fixed_string<32>(view.residual_correction_iteration_name(), "residual_correction_iteration");
            snapshot.num_cheap_to_jd_switches   = static_cast<int64_t>(view.cheap_to_jd_switch_outer_iters().size());
            snapshot.num_jd_to_cheap_switches   = static_cast<int64_t>(view.jd_to_cheap_switch_outer_iters().size());
            snapshot.cheap_to_jd_switch_outer_iters   = to_i64_vector(view.cheap_to_jd_switch_outer_iters());
            snapshot.jd_to_cheap_switch_outer_iters   = to_i64_vector(view.jd_to_cheap_switch_outer_iters());
            return snapshot;
        }

        template<typename Solver>
        auto make_solver_snapshot(const Options &opts, int rep, unsigned int rep_seed, const Solver &, grit::ResultView<Scalar> view)
            -> std::conditional_t<std::is_same_v<Solver, LanczosSolver>, LanczosSnapshot, LobpcgSnapshot> {
            using Snapshot = std::conditional_t<std::is_same_v<Solver, LanczosSolver>, LanczosSnapshot, LobpcgSnapshot>;
            Snapshot snapshot;
            fill_common_snapshot(snapshot, opts, rep, rep_seed, view);
            if constexpr(std::is_same_v<Solver, LanczosSolver>) {
                snapshot.max_retain_blocks      = static_cast<int32_t>(opts.maxRetainBlocks);
                snapshot.time_orthogonalize  = view.time_orthogonalize();
                snapshot.time_orthonormalize = view.time_orthonormalize();
                snapshot.time_orth_project   = view.time_orth_project();
                snapshot.time_orth_factor    = view.time_orth_factor();
                snapshot.time_orth_update    = view.time_orth_update();
                snapshot.time_orth_refresh   = view.time_orth_refresh();
                snapshot.time_orth_mask      = view.time_orth_mask();
                snapshot.time_diagonalize    = view.time_diagonalize();
                snapshot.time_extract_ritz   = view.time_extract_ritz();
                snapshot.time_restart        = view.time_restart();
            }
            return snapshot;
        }

        template<typename Solver>
        struct result_for;

        template<>
        struct result_for<GdSolver> {
            using type = GdpluskSolveResult;
        };

        template<>
        struct result_for<LanczosSolver> {
            using type = LanczosSolveResult;
        };

        template<>
        struct result_for<LobpcgSolver> {
            using type = LobpcgSolveResult;
        };

        template<typename Solver>
        auto solve_once_impl(const SparseMatrix &matrix_a, const SparseMatrix &matrix_b, Options opts, int rep)
            -> typename result_for<Solver>::type {
            using Result = typename result_for<Solver>::type;

            auto Aop = grit::matvec<Scalar>(matrix_a.rows(), [&](const DenseMatrixRef &X) -> DenseMatrix { return matrix_a * X; });
            auto Bop = grit::matvec<Scalar>(matrix_b.rows(), [&](const DenseMatrixRef &X) -> DenseMatrix { return matrix_b * X; });

            auto guess = initial_guess(opts, matrix_a.rows(), rep);
            Solver solver(Aop, Bop);

            solver.config.nev                          = opts.nev;
            solver.config.ncv                          = opts.ncv;
            solver.config.block_size                   = opts.block_size;
            solver.config.ritz                         = opts.ritz;
            solver.config.use_refined_rayleigh_ritz    = opts.use_refined_rayleigh_ritz;
            solver.config.use_b_inner_product          = opts.use_b_inner_product;
            solver.config.use_rescaled_rnorm_tolerance = opts.use_rescaled_rnorm_tolerance;
            solver.config.max_iters                    = opts.max_iters;
            solver.config.max_matvecs                  = opts.max_matvecs;
            solver.config.abstol                       = opts.abstol;
            solver.config.reltol                       = opts.reltol;
            solver.config.sat_eigval_threshold         = opts.sat_eigval_threshold;
            solver.config.sat_rnorm_threshold          = opts.sat_rnorm_threshold;
            solver.config.log_level                    = opts.log_level;

            if constexpr(std::is_same_v<Solver, GdSolver>) {
                solver.config.residual_correction_type      = opts.residual_correction;
                solver.config.use_jd_b_only                 = opts.use_jd_b_only;
                solver.config.use_adaptive_inner_tolerance  = opts.use_adaptive_inner_tolerance;
                solver.config.inner_max_iters               = opts.inner_max_iters;
                solver.config.inner_tol                     = opts.inner_tol;
                solver.config.auto_min_dwell_iters          = opts.auto_min_dwell_iters;
                solver.config.auto_sat_eigval_threshold     = opts.auto_sat_eigval_threshold;
                solver.config.auto_sat_rnorm_threshold      = opts.auto_sat_rnorm_threshold;
                solver.config.auto_jd_start_rnorm_threshold = opts.auto_jd_start_rnorm_threshold;
                solver.config.auto_cheap_probe_interval     = opts.auto_cheap_probe_interval;
                solver.config.auto_cheap_probe_factor       = opts.auto_cheap_probe_factor;
            } else if constexpr(std::is_same_v<Solver, LanczosSolver>) {
                solver.config.maxRetainBlocks = opts.maxRetainBlocks;
            }

            solver.set_initial_guess(guess);

            const auto time_start = std::chrono::steady_clock::now();
            const auto rep_seed   = opts.seed + static_cast<unsigned int>(rep - 1);
            std::vector<decltype(make_solver_snapshot(opts, rep, rep_seed, solver, solver.get_result_view()))> snapshots;
            solver.config.user_callback = [&](const Solver &solver_ref) {
                snapshots.push_back(make_solver_snapshot(opts, rep, rep_seed, solver_ref, solver_ref.get_result_view()));
            };
            solver.run();
            const auto time_stop = std::chrono::steady_clock::now();

            const auto view = solver.get_result_view();
            auto final_snapshot = make_solver_snapshot(opts, rep, rep_seed, solver, view);
            final_snapshot.time   = std::chrono::duration<double>(time_stop - time_start).count();
            final_snapshot.vmrss_mib = mem_usage_in_mib("VmRSS");
            final_snapshot.vmhwm_mib = mem_usage_in_mib("VmHWM");
            final_snapshot.vmpeak_mib = mem_usage_in_mib("VmPeak");
            if(snapshots.empty())
                snapshots.push_back(final_snapshot);
            else
                snapshots.back() = final_snapshot;

            Result solve_result{
                .final     = final_snapshot,
                .snapshots = std::move(snapshots),
            };
            if(view.eigVecs().cols() >= opts.nev) solve_result.eigvecs = view.eigVecs().leftCols(opts.nev);
            return solve_result;
        }
    }

    SolveResult solve_once(const SparseMatrix &matrix_a, const SparseMatrix &matrix_b, Options opts, int rep) {
        switch(opts.algo) {
            case Algo::gdplusk: return solve_once_impl<GdSolver>(matrix_a, matrix_b, opts, rep);
            case Algo::lanczos: return solve_once_impl<LanczosSolver>(matrix_a, matrix_b, opts, rep);
            case Algo::lobpcg: return solve_once_impl<LobpcgSolver>(matrix_a, matrix_b, opts, rep);
        }
        throw std::runtime_error("unsupported benchmark algorithm");
    }
}

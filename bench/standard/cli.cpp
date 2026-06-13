#include "cli.h"
#include "text.h"
#include <algorithm>
#include <format>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace bench_standard {
    namespace {
        std::vector<std::string> split_list(std::string text, std::string_view name) {
            text = trim_copy(std::move(text));
            if(text.size() >= 2 && text.front() == '[' && text.back() == ']') text = text.substr(1, text.size() - 2);
            text = trim_copy(std::move(text));
            if(text.empty()) throw std::runtime_error(std::format("{} must not be empty", name));

            std::vector<std::string> values;
            std::stringstream        stream(text);
            std::string              item;
            while(std::getline(stream, item, ',')) {
                item = trim_copy(std::move(item));
                if(item.empty()) throw std::runtime_error(std::format("{} contains an empty list item", name));
                values.push_back(std::move(item));
            }
            return values;
        }

        grit::Ritz parse_ritz(std::string text) {
            text = lower_copy(trim_copy(std::move(text)));
            if(text == "none") return grit::Ritz::NONE;
            if(text == "lr") return grit::Ritz::LR;
            if(text == "lm") return grit::Ritz::LM;
            if(text == "sr") return grit::Ritz::SR;
            if(text == "sm") return grit::Ritz::SM;
            throw std::runtime_error(std::format("--ritz has invalid value '{}'", text));
        }

        ResidualCorrection parse_residual_correction(std::string text) {
            text = lower_copy(trim_copy(std::move(text)));
            if(text == "none") return ResidualCorrection::NONE;
            if(text == "cheap-olsen") return ResidualCorrection::CHEAP_OLSEN;
            if(text == "full-olsen") return ResidualCorrection::FULL_OLSEN;
            if(text == "jacobi-davidson") return ResidualCorrection::JACOBI_DAVIDSON;
            if(text == "auto") return ResidualCorrection::AUTO;
            throw std::runtime_error(std::format("--residual-correction has invalid value '{}'", text));
        }

        template<typename T, typename Parser>
        std::vector<T> parse_list_as(const std::string &text, std::string_view name, Parser parser) {
            auto           items = split_list(text, name);
            std::vector<T> values;
            values.reserve(items.size());
            for(auto &item : items) values.push_back(parser(std::move(item)));
            return values;
        }

        template<typename T, typename Predicate>
        void require_all(const std::vector<T> &values, std::string_view name, Predicate predicate, std::string_view message) {
            if(values.empty()) throw std::runtime_error(std::format("{} must not be empty", name));
            for(const auto &value : values)
                if(!predicate(value)) throw std::runtime_error(std::format("{} {}", name, message));
        }

        std::string strip_bracket_list(std::string text) {
            text = trim_copy(std::move(text));
            if(text.size() >= 2 && text.front() == '[' && text.back() == ']') return text.substr(1, text.size() - 2);
            return text;
        }

        void adjust_block_geometry(Options &opts, int requested_ncv, int requested_block_size, Eigen::Index matrix_rows) {
            const auto rows = static_cast<int>(matrix_rows);
            opts.nev        = std::min(std::max(1, opts.nev), rows);
            opts.block_size = std::clamp(std::max(opts.nev, requested_block_size), 1, rows);
            opts.ncv        = std::min(std::max(opts.block_size, requested_ncv), rows);
            opts.ncv        = std::max(opts.block_size, (opts.ncv / opts.block_size) * opts.block_size);
        }

        void validate_algo_specific_options(const CliOptions &opts) {
            auto reject = [&](bool explicit_option, std::string_view option_group) {
                if(explicit_option) {
                    throw std::runtime_error(std::format("--{} is only supported for --algo=gdplusk", option_group));
                }
            };

            if(opts.algo != Algo::gdplusk) {
                reject(opts.explicit_residual_correction, "residual-correction");
                reject(opts.explicit_inner_max_iters, "inner-max-iters");
                reject(opts.explicit_inner_tol, "inner-tol");
                reject(opts.explicit_use_adaptive_inner_tolerance, "use-adaptive-inner-tolerance");
                reject(opts.explicit_auto_min_dwell_iters, "auto-min-dwell-iters");
                reject(opts.explicit_auto_sat_eigval_threshold, "auto-sat-eigval-threshold");
                reject(opts.explicit_auto_sat_rnorm_threshold, "auto-sat-rnorm-threshold");
                reject(opts.explicit_auto_jd_start_rnorm_threshold, "auto-jd-start-rnorm-threshold");
                reject(opts.explicit_auto_cheap_probe_interval, "auto-cheap-probe-interval");
                reject(opts.explicit_auto_cheap_probe_factor, "auto-cheap-probe-factor");
            }

            if(opts.algo != Algo::lanczos && opts.explicit_max_retain_blocks) {
                throw std::runtime_error("--max-retain-blocks is only supported for --algo=lanczos");
            }
        }
    }

    std::vector<std::string> normalized_cli_args(int argc, char **argv) {
        std::vector<std::string> args;
        args.reserve(static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
        for(auto i = argc - 1; i > 0; --i) {
            std::string arg = argv[i];
            if(arg == "--refined-rayleigh-ritz") arg = "--refined-rayleigh-ritz=true";
            if(arg == "--use-adaptive-inner-tolerance") arg = "--use-adaptive-inner-tolerance=true";
            auto eq = arg.find('=');
            if(eq == std::string::npos) {
                arg = strip_bracket_list(std::move(arg));
            } else {
                auto name  = arg.substr(0, eq + 1);
                auto value = strip_bracket_list(arg.substr(eq + 1));
                arg        = std::move(name) + std::move(value);
            }
            args.push_back(std::move(arg));
        }
        return args;
    }

    void configure_cli(CLI::App &app, CliOptions &opts) {
        const std::map<std::string, spdlog::level::level_enum, std::less<>> log_level_map{
            {"trace", spdlog::level::trace}, {"debug", spdlog::level::debug},       {"info", spdlog::level::info}, {"warn", spdlog::level::warn},
            {"err", spdlog::level::err},     {"critical", spdlog::level::critical}, {"off", spdlog::level::off},
        };
        const std::map<std::string, Algo, std::less<>> algo_map{
            {"gdplusk", Algo::gdplusk},
            {"lanczos", Algo::lanczos},
            {"lobpcg", Algo::lobpcg},
        };

        app.option_defaults()->always_capture_default();
        app.allow_extras(false);
        /* clang-format off */
        app.add_option("--algo", opts.algo, "Solver algorithm [gdplusk, lanczos, lobpcg]")->transform(CLI::CheckedTransformer(algo_map, CLI::ignore_case))->type_name("ENUM");
        app.add_option("--matrix-path", opts.matrix_path, "Path to a Matrix Market .mtx file")->check(CLI::ExistingFile);
        app.add_option("--initial-guess", opts.initial_guess, "Path to HDF5 file with /grit/standard/<algo>/eigvecs initial guess (legacy /grit/standard/eigvecs also accepted)")->check(CLI::ExistingFile);
        app.add_option("--save-eigvec", opts.save_eigvec, "Path to HDF5 file where final eigenvectors are saved");
        app.add_option("--save-results", opts.save_results, "Path to HDF5 file where final result rows and solver snapshots are saved");
        app.add_flag("--print-summary", opts.print_summary, "Print the summary from --save-results without running the benchmark");
        app.add_option("--nev", opts.nev, "Number of eigenpairs")->check(CLI::PositiveNumber);
        app.add_option("--ncv", opts.ncv, "Maximum subspace columns, or a comma list like [8,16]")->delimiter(',');
        app.add_option("--block-size", opts.block_size, "Solver block size, or a comma list like [1,2]")->delimiter(',');
        app.add_option("--max-iters", opts.max_iters, "Maximum solver outer iterations, or a negative value for unlimited");
        app.add_option("--max-matvecs", opts.max_matvecs, "Maximum matrix-vector products, or a negative value for unlimited");
        auto *inner_max_iters_opt = app.add_option("--inner-max-iters", opts.inner_max_iters, "Maximum Jacobi-Davidson inner iterations, or a comma list")->delimiter(',');
        auto *max_retain_blocks_opt = app.add_option("--max-retain-blocks", opts.max_retain_blocks, "Lanczos retained restart blocks, or a comma list")->delimiter(',');
        app.add_option("--reps", opts.reps, "Number of benchmark repetitions")->check(CLI::PositiveNumber);
        app.add_option("--tol", opts.tol, "Absolute residual-norm convergence tolerance, or relative residual tolerance with --use-relative-rnorm-tolerance")->delimiter(',');
        app.add_option("--tol-rnorm-relative", opts.tol_rnorm_relative, "Relative-to-initial absolute residual-norm convergence tolerance");
        app.add_option("--sat-eigval-threshold", opts.sat_eigval_threshold, "Stop if eigenvalue-history relative standard deviation is below this tolerance; 0 disables it");
        app.add_option("--sat-rnorm-threshold", opts.sat_rnorm_threshold,
                       "Stop if derived relative-residual history standard deviation is below this fraction of the current derived relative residual; 0 disables it");
        auto *inner_tol_opt = app.add_option("--inner-tol", opts.inner_tol, "Jacobi-Davidson inner tolerance, or a comma list")->delimiter(',');
        auto *auto_min_dwell_iters_opt = app.add_option("--auto-min-dwell-iters", opts.auto_min_dwell_iters, "Minimum consecutive cheap-Olsen AUTO outer iterations before Jacobi-Davidson activation may be scheduled")->check(CLI::NonNegativeNumber);
        auto *auto_sat_eigval_threshold_opt = app.add_option("--auto-sat-eigval-threshold", opts.auto_sat_eigval_threshold, "AUTO eigenvalue-history relative standard deviation tolerance");
        auto *auto_sat_rnorm_threshold_opt = app.add_option("--auto-sat-rnorm-threshold", opts.auto_sat_rnorm_threshold, "AUTO derived-relative-residual saturation tolerance");
        auto *auto_jd_start_rnorm_threshold_opt = app.add_option("--auto-jd-start-rnorm-threshold", opts.auto_jd_start_rnorm_threshold, "Derived relative residual norm below which AUTO may activate JD; 0 disables it");
        auto *auto_cheap_probe_interval_opt = app.add_option("--auto-cheap-probe-interval", opts.auto_cheap_probe_interval, "Jacobi-Davidson outer iterations before AUTO forces a cheap-Olsen probe")->check(CLI::PositiveNumber);
        auto *auto_cheap_probe_factor_opt = app.add_option("--auto-cheap-probe-factor", opts.auto_cheap_probe_factor,
                       "Cheap probe must improve the Ritz value by this factor times max(absolute rnorm^2, roundoff scale)");
        app.add_option("--seed", opts.seed, "Random seed for deterministic initial guess");
        app.add_option("--ritz", opts.ritz, "Ritz target [SR, LR, SM, LM], or a comma list")->type_name("ENUM");
        app.add_option("--log-level", opts.log_level, "Solver log level [trace, debug, info, warn, err, critical, off]")->transform(CLI::CheckedTransformer(log_level_map, CLI::ignore_case))->type_name("ENUM");
        auto *residual_correction_opt = app.add_option("--residual-correction", opts.residual_correction,"Residual correction [none, cheap-olsen, full-olsen, jacobi-davidson, auto], or a comma list")->type_name("ENUM");
        app.add_option("--refined-rayleigh-ritz", opts.use_refined_rayleigh_ritz, "Enable refined Rayleigh-Ritz extraction, or use [false,true]")->delimiter(',');
        app.add_flag("--use-relative-rnorm-tolerance", opts.use_relative_rnorm_tolerance, "Enable relative residual-norm tolerance");
        auto *adaptive_inner_tolerance_opt = app.add_option("--use-adaptive-inner-tolerance", opts.use_adaptive_inner_tolerance, "Enable adaptive inner tolerance, or use [true,false]")->delimiter(',');
        /* clang-format off */

        app.callback([&opts, inner_max_iters_opt, max_retain_blocks_opt, inner_tol_opt, auto_min_dwell_iters_opt, auto_sat_eigval_threshold_opt, auto_sat_rnorm_threshold_opt,
                      auto_jd_start_rnorm_threshold_opt, auto_cheap_probe_interval_opt, auto_cheap_probe_factor_opt, residual_correction_opt,
                      adaptive_inner_tolerance_opt]() {
            opts.explicit_inner_max_iters               = inner_max_iters_opt->count() > 0;
            opts.explicit_max_retain_blocks             = max_retain_blocks_opt->count() > 0;
            opts.explicit_inner_tol                     = inner_tol_opt->count() > 0;
            opts.explicit_auto_min_dwell_iters          = auto_min_dwell_iters_opt->count() > 0;
            opts.explicit_auto_sat_eigval_threshold     = auto_sat_eigval_threshold_opt->count() > 0;
            opts.explicit_auto_sat_rnorm_threshold      = auto_sat_rnorm_threshold_opt->count() > 0;
            opts.explicit_auto_jd_start_rnorm_threshold = auto_jd_start_rnorm_threshold_opt->count() > 0;
            opts.explicit_auto_cheap_probe_interval     = auto_cheap_probe_interval_opt->count() > 0;
            opts.explicit_auto_cheap_probe_factor       = auto_cheap_probe_factor_opt->count() > 0;
            opts.explicit_residual_correction           = residual_correction_opt->count() > 0;
            opts.explicit_use_adaptive_inner_tolerance  = adaptive_inner_tolerance_opt->count() > 0;
        });
    }

    void normalize_options(CliOptions &opts) {
        require_all(opts.ncv, "--ncv", [](int value) { return value > 0; }, "must be positive");
        require_all(opts.block_size, "--block-size", [](int value) { return value > 0; }, "must be positive");
        require_all(opts.max_retain_blocks, "--max-retain-blocks", [](int value) { return value > 0; }, "must be positive");
        require_all(opts.inner_max_iters, "--inner-max-iters", [](int value) { return value > 0; }, "must be positive");
        require_all(opts.tol, "--tol", [](double value) { return value > 0.0; }, "must be positive");
        require_all(opts.inner_tol, "--inner-tol", [](double value) { return value > 0.0; }, "must be positive");
        if(opts.use_refined_rayleigh_ritz.empty()) throw std::runtime_error("--refined-rayleigh-ritz must not be empty");
        if(opts.use_adaptive_inner_tolerance.empty()) throw std::runtime_error("--use-adaptive-inner-tolerance must not be empty");
        if(opts.tol_rnorm_relative < 0.0) throw std::runtime_error("--tol-rnorm-relative must be non-negative");
        if(opts.sat_eigval_threshold < 0.0) throw std::runtime_error("--sat-eigval-threshold must be non-negative");
        if(opts.sat_rnorm_threshold < 0.0) throw std::runtime_error("--sat-rnorm-threshold must be non-negative");
        if(opts.auto_sat_eigval_threshold < 0.0) throw std::runtime_error("--auto-sat-eigval-threshold must be non-negative");
        if(opts.auto_sat_rnorm_threshold < 0.0) throw std::runtime_error("--auto-sat-rnorm-threshold must be non-negative");
        if(opts.auto_jd_start_rnorm_threshold < 0.0) throw std::runtime_error("--auto-jd-start-rnorm-threshold must be non-negative");
        if(opts.auto_cheap_probe_interval <= 0) throw std::runtime_error("--auto-cheap-probe-interval must be positive");
        if(opts.auto_cheap_probe_factor < 0.0) throw std::runtime_error("--auto-cheap-probe-factor must be non-negative");
        if(opts.tol_rnorm_relative > 0.0) opts.use_relative_rnorm_tolerance = true;
        validate_algo_specific_options(opts);
    }

    std::vector<Options> expand_sweep(const CliOptions &cli, Eigen::Index matrix_rows) {
        const auto &ncv_values                 = cli.ncv;
        const auto &block_values               = cli.block_size;
        const auto &max_retain_block_values    = cli.max_retain_blocks;
        const auto &inner_iter_values          = cli.inner_max_iters;
        const auto &tol_values                 = cli.tol;
        const auto &inner_tol_values           = cli.inner_tol;
        const auto  ritz_values                = parse_list_as<grit::Ritz>(cli.ritz, "--ritz", [](std::string item) { return parse_ritz(std::move(item)); });
        const auto  residual_correction_values = parse_list_as<ResidualCorrection>(cli.residual_correction, "--residual-correction",
                                                                                   [](std::string item) { return parse_residual_correction(std::move(item)); });
        const auto &refined_rayleigh_ritz_values    = cli.use_refined_rayleigh_ritz;
        const auto &adaptive_inner_tolerance_values = cli.use_adaptive_inner_tolerance;

        std::vector<Options> cases;
        int                  case_id = 1;
        for(auto ncv : ncv_values)
            for(auto block_size : block_values)
                for(auto max_retain_blocks : max_retain_block_values)
                    for(auto inner_iters : inner_iter_values)
                    for(auto tol : tol_values)
                        for(auto inner_tol : inner_tol_values)
                            for(auto ritz : ritz_values)
                                for(auto residual_correction : residual_correction_values)
                                    for(auto use_refined_rayleigh_ritz : refined_rayleigh_ritz_values)
                                        for(auto use_adaptive_inner_tolerance : adaptive_inner_tolerance_values) {
                                            Options opts;
                                            opts.case_id       = case_id++;
                                            opts.algo          = cli.algo;
                                            opts.matrix_path   = cli.matrix_path;
                                            opts.initial_guess = cli.initial_guess;
                                            opts.save_eigvec   = cli.save_eigvec;
                                            opts.save_results  = cli.save_results;
                                            opts.nev           = cli.nev;
                                            adjust_block_geometry(opts, ncv, block_size, matrix_rows);
                                            opts.maxRetainBlocks             = max_retain_blocks;
                                            opts.max_iters                    = cli.max_iters;
                                            opts.max_matvecs                  = cli.max_matvecs;
                                            opts.inner_max_iters              = inner_iters;
                                            opts.reps                         = cli.reps;
                                            opts.tol                          = tol;
                                            opts.tol_rnorm_relative           = cli.tol_rnorm_relative;
                                            opts.sat_eigval_threshold         = cli.sat_eigval_threshold;
                                            opts.sat_rnorm_threshold          = cli.sat_rnorm_threshold;
                                            opts.inner_tol                    = inner_tol;
                                            opts.auto_min_dwell_iters         = cli.auto_min_dwell_iters;
                                            opts.auto_sat_eigval_threshold    = cli.auto_sat_eigval_threshold;
                                            opts.auto_sat_rnorm_threshold     = cli.auto_sat_rnorm_threshold;
                                            opts.auto_jd_start_rnorm_threshold = cli.auto_jd_start_rnorm_threshold;
                                            opts.auto_cheap_probe_interval    = cli.auto_cheap_probe_interval;
                                            opts.auto_cheap_probe_factor      = cli.auto_cheap_probe_factor;
                                            opts.seed                         = cli.seed;
                                            opts.ritz                         = ritz;
                                            opts.log_level                    = cli.log_level;
                                            opts.residual_correction          = residual_correction;
                                            opts.use_refined_rayleigh_ritz        = use_refined_rayleigh_ritz;
                                            opts.use_relative_rnorm_tolerance = cli.use_relative_rnorm_tolerance;
                                            opts.use_adaptive_inner_tolerance = use_adaptive_inner_tolerance;
                                            cases.push_back(opts);
                                        }
        return cases;
    }

    void validate_hdf5_options(const CliOptions &cli, std::size_t case_count) {
        if(cli.print_summary && cli.save_results.empty()) throw std::runtime_error("--print-summary requires --save-results=<file>");
        if(cli.save_eigvec.empty()) return;
        if(cli.reps != 1) throw std::runtime_error("--save-eigvec requires --reps=1");
        if(case_count != 1) throw std::runtime_error("--save-eigvec requires exactly one expanded sweep case");
    }
}

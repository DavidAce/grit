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

        int parse_int(std::string text, std::string_view name, bool positive) {
            text            = trim_copy(std::move(text));
            std::size_t pos = 0;
            int         value;
            try {
                value = std::stoi(text, &pos);
            } catch(const std::exception &) {
                throw std::runtime_error(std::format("{} expects an integer, got '{}'", name, text));
            }
            if(pos != text.size()) throw std::runtime_error(std::format("{} expects an integer, got '{}'", name, text));
            if(positive && value <= 0) throw std::runtime_error(std::format("{} must be positive", name));
            return value;
        }

        double parse_double(std::string text, std::string_view name, bool positive) {
            text            = trim_copy(std::move(text));
            std::size_t pos = 0;
            double      value;
            try {
                value = std::stod(text, &pos);
            } catch(const std::exception &) {
                throw std::runtime_error(std::format("{} expects a floating-point value, got '{}'", name, text));
            }
            if(pos != text.size()) throw std::runtime_error(std::format("{} expects a floating-point value, got '{}'", name, text));
            if(positive && value <= 0.0) throw std::runtime_error(std::format("{} must be positive", name));
            return value;
        }

        bool parse_bool(std::string text, std::string_view name) {
            text = lower_copy(trim_copy(std::move(text)));
            if(text == "true" || text == "1" || text == "on" || text == "yes") return true;
            if(text == "false" || text == "0" || text == "off" || text == "no") return false;
            throw std::runtime_error(std::format("{} expects a boolean value, got '{}'", name, text));
        }

        grit::OptRitz parse_ritz(std::string text) {
            text = lower_copy(trim_copy(std::move(text)));
            if(text == "none") return grit::OptRitz::NONE;
            if(text == "lr") return grit::OptRitz::LR;
            if(text == "lm") return grit::OptRitz::LM;
            if(text == "sr") return grit::OptRitz::SR;
            if(text == "sm") return grit::OptRitz::SM;
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

        void adjust_block_geometry(Options &opts, int requested_ncv, int requested_block_size, int requested_max_basis_blocks, Eigen::Index matrix_rows) {
            const auto rows = static_cast<int>(matrix_rows);
            opts.nev        = std::min(std::max(1, opts.nev), rows);
            opts.block_size = std::clamp(std::max(opts.nev, requested_block_size), 1, rows);

            const auto max_basis_blocks = std::max(1, rows / opts.block_size);
            if(requested_ncv < 0) {
                opts.max_basis_blocks = std::clamp(requested_max_basis_blocks, 1, max_basis_blocks);
                opts.ncv              = opts.max_basis_blocks * opts.block_size;
            } else {
                opts.ncv              = std::min(std::max(opts.nev, requested_ncv), rows);
                opts.max_basis_blocks = std::clamp(opts.ncv / opts.block_size, 1, max_basis_blocks);
                opts.ncv              = opts.max_basis_blocks * opts.block_size;
            }
        }
    }

    std::vector<std::string> normalized_cli_args(int argc, char **argv) {
        std::vector<std::string> args;
        args.reserve(static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
        for(auto i = argc - 1; i > 0; --i) {
            std::string arg = argv[i];
            if(arg == "--refined-rayleigh-ritz") arg = "--refined-rayleigh-ritz=true";
            if(arg == "--adaptive-inner-tolerance") arg = "--adaptive-inner-tolerance=true";
            args.push_back(std::move(arg));
        }
        return args;
    }

    void configure_cli(CLI::App &app, CliOptions &opts) {
        const std::map<std::string, spdlog::level::level_enum, std::less<>> log_level_map{
            {"trace", spdlog::level::trace}, {"debug", spdlog::level::debug},       {"info", spdlog::level::info},
            {"warn", spdlog::level::warn},   {"err", spdlog::level::err},           {"critical", spdlog::level::critical},
            {"off", spdlog::level::off},
        };

        app.add_option("--matrix", opts.matrix, "Path to a Matrix Market .mtx file")->default_val(opts.matrix)->check(CLI::ExistingFile);
        app.add_option("--initial-guess", opts.initial_guess, "Path to HDF5 file with /grit/standard/eigvecs initial guess")->check(CLI::ExistingFile);
        app.add_option("--save-eigvec", opts.save_eigvec, "Path to HDF5 file where final eigenvectors are saved");
        app.add_option("--nev", opts.nev, "Number of eigenpairs")->default_val(opts.nev)->check(CLI::PositiveNumber);
        app.add_option("--ncv", opts.ncv,
                       "Maximum subspace columns, or a comma list like [8,16]. Negative derives ncv from --max-basis-blocks * --block-size")
            ->default_val(opts.ncv);
        app.add_option("--block-size", opts.block_size, "Solver block size, or a comma list like [1,2]")->default_val(opts.block_size);
        app.add_option("--max-basis-blocks", opts.max_basis_blocks, "Number of basis blocks used when --ncv is negative")
            ->default_val(opts.max_basis_blocks)
            ->check(CLI::PositiveNumber);
        app.add_option("--max-iters", opts.max_iters, "Maximum solver iterations, or a negative value for unlimited")->default_val(opts.max_iters);
        app.add_option("--max-matvecs", opts.max_matvecs, "Maximum matrix-vector products, or a negative value for unlimited")->default_val(opts.max_matvecs);
        app.add_option("--inner-iters", opts.inner_iters, "Maximum Jacobi-Davidson inner iterations, or a comma list")->default_val(opts.inner_iters);
        app.add_option("--reps", opts.reps, "Number of benchmark repetitions")->default_val(opts.reps)->check(CLI::PositiveNumber);
        app.add_option("--tol", opts.tol, "Absolute convergence tolerance, or a comma list")->default_val(opts.tol);
        app.add_option("--reltol", opts.reltol, "Relative convergence tolerance")->default_val(opts.reltol);
        app.add_option("--tol-stall-evals", opts.tol_stall_evals,
                       "Stop if eigenvalue-history relative standard deviation is below this tolerance; 0 disables it")
            ->default_val(opts.tol_stall_evals);
        app.add_option("--tol-stall-rnorm", opts.tol_stall_rnorm,
                       "Stop if residual-history standard deviation is below this fraction of the current residual norm; 0 disables it")
            ->default_val(opts.tol_stall_rnorm);
        app.add_option("--inner-tol", opts.inner_tol, "Jacobi-Davidson inner tolerance, or a comma list")->default_val(opts.inner_tol);
        app.add_option("--seed", opts.seed, "Random seed for deterministic initial guess")->default_val(opts.seed);
        app.add_option("--ritz", opts.ritz, "Ritz target [sr, lr, sm, lm], or a comma list")->default_val(opts.ritz)->type_name("ENUM");
        app.add_option("--log-level", opts.log_level, "Solver log level [trace, debug, info, warn, err, critical, off]")
            ->transform(CLI::CheckedTransformer(log_level_map, CLI::ignore_case))
            ->type_name("ENUM");
        app.add_option("--residual-correction", opts.residual_correction,
                       "Residual correction [none, cheap-olsen, full-olsen, jacobi-davidson, auto], or a comma list")
            ->default_val(opts.residual_correction)
            ->type_name("ENUM");
        app.add_option("--refined-rayleigh-ritz", opts.refined_rayleigh_ritz, "Enable refined Rayleigh-Ritz extraction, or use [false,true]")
            ->default_val(opts.refined_rayleigh_ritz)
            ->expected(0, 1);
        app.add_flag("--relative-rnorm-tolerance", opts.relative_rnorm_tolerance, "Enable relative residual-norm tolerance");
        app.add_option("--adaptive-inner-tolerance", opts.adaptive_inner_tolerance, "Enable adaptive inner tolerance, or use [true,false]")
            ->default_val(opts.adaptive_inner_tolerance)
            ->expected(0, 1);
    }

    void normalize_options(CliOptions &opts) {
        if(opts.reltol < 0.0) throw std::runtime_error("--reltol must be non-negative");
        if(opts.tol_stall_evals < 0.0) throw std::runtime_error("--tol-stall-evals must be non-negative");
        if(opts.tol_stall_rnorm < 0.0) throw std::runtime_error("--tol-stall-rnorm must be non-negative");
        if(opts.reltol > 0.0) opts.relative_rnorm_tolerance = true;
    }

    std::vector<Options> expand_sweep(const CliOptions &cli, Eigen::Index matrix_rows) {
        const auto ncv_values = parse_list_as<int>(cli.ncv, "--ncv", [](std::string item) {
            auto value = parse_int(std::move(item), "--ncv", false);
            if(value == 0) throw std::runtime_error("--ncv must be positive, or negative to derive it from --max-basis-blocks * --block-size");
            return value;
        });
        const auto block_values =
            parse_list_as<int>(cli.block_size, "--block-size", [](std::string item) { return parse_int(std::move(item), "--block-size", true); });
        const auto inner_iter_values =
            parse_list_as<int>(cli.inner_iters, "--inner-iters", [](std::string item) { return parse_int(std::move(item), "--inner-iters", true); });
        const auto tol_values = parse_list_as<double>(cli.tol, "--tol", [](std::string item) { return parse_double(std::move(item), "--tol", true); });
        const auto inner_tol_values =
            parse_list_as<double>(cli.inner_tol, "--inner-tol", [](std::string item) { return parse_double(std::move(item), "--inner-tol", true); });
        const auto ritz_values =
            parse_list_as<grit::OptRitz>(cli.ritz, "--ritz", [](std::string item) { return parse_ritz(std::move(item)); });
        const auto residual_correction_values = parse_list_as<ResidualCorrection>(
            cli.residual_correction, "--residual-correction", [](std::string item) { return parse_residual_correction(std::move(item)); });
        const auto refined_rayleigh_ritz_values =
            parse_list_as<bool>(cli.refined_rayleigh_ritz.empty() ? "true" : cli.refined_rayleigh_ritz, "--refined-rayleigh-ritz",
                                [](std::string item) { return parse_bool(std::move(item), "--refined-rayleigh-ritz"); });
        const auto adaptive_inner_tolerance_values = parse_list_as<bool>(
            cli.adaptive_inner_tolerance.empty() ? "true" : cli.adaptive_inner_tolerance, "--adaptive-inner-tolerance",
            [](std::string item) { return parse_bool(std::move(item), "--adaptive-inner-tolerance"); });

        std::vector<Options> cases;
        int                  case_id = 1;
        for(auto ncv : ncv_values)
            for(auto block_size : block_values)
                for(auto inner_iters : inner_iter_values)
                    for(auto tol : tol_values)
                        for(auto inner_tol : inner_tol_values)
                            for(auto ritz : ritz_values)
                                for(auto residual_correction : residual_correction_values)
                                    for(auto refined_rayleigh_ritz : refined_rayleigh_ritz_values)
                                        for(auto adaptive_inner_tolerance : adaptive_inner_tolerance_values) {
                                            Options opts;
                                            opts.case_id                  = case_id++;
                                            opts.matrix                   = cli.matrix;
                                            opts.initial_guess            = cli.initial_guess;
                                            opts.save_eigvec              = cli.save_eigvec;
                                            opts.nev                      = cli.nev;
                                            adjust_block_geometry(opts, ncv, block_size, cli.max_basis_blocks, matrix_rows);
                                            opts.max_iters                = cli.max_iters;
                                            opts.max_matvecs              = cli.max_matvecs;
                                            opts.inner_iters              = inner_iters;
                                            opts.reps                     = cli.reps;
                                            opts.tol                      = tol;
                                            opts.reltol                   = cli.reltol;
                                            opts.tol_stall_evals  = cli.tol_stall_evals;
                                            opts.tol_stall_rnorm = cli.tol_stall_rnorm;
                                            opts.inner_tol                = inner_tol;
                                            opts.seed                     = cli.seed;
                                            opts.ritz                     = ritz;
                                            opts.log_level                = cli.log_level;
                                            opts.residual_correction      = residual_correction;
                                            opts.refined_rayleigh_ritz    = refined_rayleigh_ritz;
                                            opts.relative_rnorm_tolerance = cli.relative_rnorm_tolerance;
                                            opts.adaptive_inner_tolerance = adaptive_inner_tolerance;
                                            cases.push_back(opts);
                                        }
        return cases;
    }

    void validate_hdf5_options(const CliOptions &cli, std::size_t case_count) {
        if(cli.save_eigvec.empty()) return;
        if(cli.reps != 1) throw std::runtime_error("--save-eigvec requires --reps=1");
        if(case_count != 1) throw std::runtime_error("--save-eigvec requires exactly one expanded sweep case");
    }
}

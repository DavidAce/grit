#pragma once

#include "cli_options.h"
#include "options.h"
#include "solve_result.h"
#include <cstddef>
#include <string>
#include <string_view>

namespace bench_standard {
    std::string_view bool_text(bool value);
    std::string_view residual_correction_name(ResidualCorrection correction);
    std::string      limit_text(Eigen::Index value);
    std::string_view log_level_name(spdlog::level::level_enum level);
    void             print_solver_config(const Options &opts, const Solver::Config &cfg);
    void             print_sweep_config(const CliOptions &opts, std::size_t cases);
    void             print_result_header();
    void             print_result_row(const SolveResult &result);
}

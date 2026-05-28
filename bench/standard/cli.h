#pragma once

#include "cli_options.h"
#include "options.h"
#include <CLI/CLI.hpp>
#include <cstddef>
#include <Eigen/Core>
#include <string>
#include <vector>

namespace bench_standard {
    std::vector<std::string> normalized_cli_args(int argc, char **argv);
    void                     configure_cli(CLI::App &app, CliOptions &opts);
    void                     normalize_options(CliOptions &opts);
    std::vector<Options>     expand_sweep(const CliOptions &cli, Eigen::Index matrix_rows);
    void                     validate_hdf5_options(const CliOptions &cli, std::size_t case_count);
}

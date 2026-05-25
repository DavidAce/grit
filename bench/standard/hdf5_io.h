#pragma once

#include "aliases.h"
#include "options.h"
#include "solve_result.h"

#include <filesystem>

namespace bench_standard {
    DenseMatrix load_initial_guess_hdf5(const std::filesystem::path &path);
    void        save_eigvecs_hdf5(const std::filesystem::path &path, const Options &opts, const SolveResult &result);
}

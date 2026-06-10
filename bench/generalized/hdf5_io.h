#pragma once

#include "aliases.h"
#include "options.h"
#include "solve_result.h"
#include <filesystem>

namespace bench_generalized {
    DenseMatrix load_initial_guess_hdf5(const std::filesystem::path &path, Algo algo);
    void        save_eigvecs_hdf5(const std::filesystem::path &path, const SolveResult &result);
    void        initialize_results_hdf5(const std::filesystem::path &path, Algo algo);
    void        append_result_hdf5(const std::filesystem::path &path, const SolveResult &result);
    void        print_results_summary_hdf5(const std::filesystem::path &path, Algo algo);
}

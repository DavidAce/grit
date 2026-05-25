#include "cli.h"
#include "hdf5_io.h"
#include "matrix_market.h"
#include "report.h"
#include "runner.h"

#include <CLI/CLI.hpp>
#include <cstdio>
#include <filesystem>
#include <print>
#include <stdexcept>

int main(int argc, char **argv) {
    bench_standard::CliOptions opts;
    CLI::App                   app{"GRIT standard eigenvalue benchmark"};
    if(argc > 0) app.name(argv[0]);
    bench_standard::configure_cli(app, opts);

    try {
        auto args = bench_standard::normalized_cli_args(argc, argv);
        app.parse(args);
    } catch(const CLI::ParseError &e) {
        return app.exit(e);
    }

    try {
        bench_standard::normalize_options(opts);
        const std::filesystem::path matrix_path = opts.matrix;
        auto                        matrix      = bench_standard::read_matrix_market(matrix_path);
        auto                        cases       = bench_standard::expand_sweep(opts, matrix.rows());
        bench_standard::validate_hdf5_options(opts, cases.size());

        std::println("matrix: {} ({})", opts.matrix, matrix_path.string());
        std::println("shape: {} x {} | nonzeros: {} | nev: {}", matrix.rows(), matrix.cols(), matrix.nonZeros(), opts.nev);
        bench_standard::print_sweep_config(opts, cases.size());
        bench_standard::print_result_header();

        for(const auto &case_opts : cases) {
            for(int rep = 1; rep <= opts.reps; ++rep) {
                const auto result = bench_standard::solve_once(matrix, case_opts, rep);
                bench_standard::print_result_row(result);
                if(!case_opts.save_eigvec.empty()) bench_standard::save_eigvecs_hdf5(case_opts.save_eigvec, case_opts, result);
            }
        }
    } catch(const std::exception &ex) {
        std::println(stderr, "grit-bench-standard: {}", ex.what());
        return 1;
    }

    return 0;
}

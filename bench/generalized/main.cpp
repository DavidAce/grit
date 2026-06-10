#include "cli.h"
#include "hdf5_io.h"
#include "matrix_market.h"
#include "report.h"
#include "runner.h"
#include <CLI/CLI.hpp>
#include <cstdio>
#include <filesystem>
#include <format>
#include <print>
#include <stdexcept>

int main(int argc, char **argv) {
    bench_generalized::CliOptions opts;
    CLI::App                      app{"GRIT generalized eigenvalue benchmark"};
    if(argc > 0) app.name(argv[0]);
    bench_generalized::configure_cli(app, opts);

    try {
        auto args = bench_generalized::normalized_cli_args(argc, argv);
        app.parse(args);
    } catch(const CLI::ParseError &e) { return app.exit(e); }

    try {
        bench_generalized::normalize_options(opts);
        if(opts.print_summary) {
            if(opts.save_results.empty()) throw std::runtime_error("--print-summary requires --save-results=<file>");
            bench_generalized::print_results_summary_hdf5(opts.save_results, opts.algo);
            return 0;
        }

        const std::filesystem::path matrix_a_path = opts.matrix_a_path;
        const std::filesystem::path matrix_b_path = opts.matrix_b_path;
        auto                        matrix_a      = bench_generalized::read_matrix_market(matrix_a_path);
        auto                        matrix_b      = bench_generalized::read_matrix_market(matrix_b_path);
        if(matrix_a.rows() != matrix_b.rows() || matrix_a.cols() != matrix_b.cols()) {
            throw std::runtime_error(std::format("Generalized benchmark requires A and B with matching dimensions: A is {} x {}, B is {} x {}", matrix_a.rows(),
                                                 matrix_a.cols(), matrix_b.rows(), matrix_b.cols()));
        }
        auto cases = bench_generalized::expand_sweep(opts, matrix_a.rows());
        bench_generalized::validate_hdf5_options(opts, cases.size());
        if(!opts.save_results.empty()) bench_generalized::initialize_results_hdf5(opts.save_results, opts.algo);

        std::println("matrix A: {} ({})", opts.matrix_a_path, matrix_a_path.string());
        std::println("matrix B: {} ({})", opts.matrix_b_path, matrix_b_path.string());
        std::println("shape: {} x {} | nonzeros A: {} | nonzeros B: {} | nev: {} | algo: {}", matrix_a.rows(), matrix_a.cols(), matrix_a.nonZeros(),
                     matrix_b.nonZeros(), opts.nev, bench_generalized::algo_name(opts.algo));
        bench_generalized::print_sweep_config(opts, cases.size());
        bench_generalized::print_result_header(opts.algo);

        for(const auto &case_opts : cases) {
            for(int rep = 1; rep <= opts.reps; ++rep) {
                const auto result = bench_generalized::solve_once(matrix_a, matrix_b, case_opts, rep);
                bench_generalized::print_result_row(result);
                if(!case_opts.save_eigvec.empty()) bench_generalized::save_eigvecs_hdf5(case_opts.save_eigvec, result);
                if(!case_opts.save_results.empty()) bench_generalized::append_result_hdf5(case_opts.save_results, result);
            }
        }

        if(!opts.save_results.empty()) bench_generalized::print_results_summary_hdf5(opts.save_results, opts.algo);
    } catch(const std::exception &ex) {
        std::println(stderr, "grit-bench-generalized: {}", ex.what());
        return 1;
    }

    return 0;
}

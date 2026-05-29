#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <Eigen/SparseCore>
#include <filesystem>
#include <format>
#include <fstream>
#include <grit/grit.h>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    using Scalar         = double;
    using DenseMatrix    = grit::Matvec<Scalar>::MatrixType;
    using DenseMatrixRef = Eigen::Ref<const DenseMatrix>;
    using SparseMatrix   = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;

    std::string lower_copy(std::string str) {
        std::ranges::transform(str, str.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return str;
    }

    SparseMatrix read_matrix_market(const std::filesystem::path &path) {
        std::ifstream file(path);
        if(!file) throw std::runtime_error(std::format("Could not open Matrix Market file: {}", path.string()));

        std::string line;
        if(!std::getline(file, line)) throw std::runtime_error(std::format("Matrix Market file is empty: {}", path.string()));

        std::istringstream header_stream(line);
        std::string        banner;
        std::string        object;
        std::string        format;
        std::string        field;
        std::string        symmetry;
        header_stream >> banner >> object >> format >> field >> symmetry;

        banner   = lower_copy(banner);
        object   = lower_copy(object);
        format   = lower_copy(format);
        field    = lower_copy(field);
        symmetry = lower_copy(symmetry);

        if(banner != "%%matrixmarket" || object != "matrix") throw std::runtime_error(std::format("Invalid Matrix Market banner in: {}", path.string()));
        if(format != "coordinate") throw std::runtime_error(std::format("Only Matrix Market coordinate format is supported: {}", path.string()));
        if(field != "real" && field != "integer" && field != "pattern") {
            throw std::runtime_error(std::format("Only real, integer, and pattern Matrix Market fields are supported: {}", path.string()));
        }
        if(symmetry != "general" && symmetry != "symmetric") {
            throw std::runtime_error(std::format("Only general and symmetric Matrix Market symmetries are supported: {}", path.string()));
        }

        do {
            if(!std::getline(file, line)) throw std::runtime_error(std::format("Missing Matrix Market size line in: {}", path.string()));
        } while(line.empty() || line.front() == '%');

        Eigen::Index rows = 0;
        Eigen::Index cols = 0;
        Eigen::Index nnz  = 0;
        {
            std::istringstream size_stream(line);
            size_stream >> rows >> cols >> nnz;
            if(!size_stream || rows <= 0 || cols <= 0 || nnz < 0)
                throw std::runtime_error(std::format("Invalid Matrix Market size line in: {}", path.string()));
        }
        if(rows != cols) throw std::runtime_error(std::format("Standard eigenvalue benchmark requires a square matrix: {}", path.string()));

        std::vector<Eigen::Triplet<Scalar>> triplets;
        triplets.reserve(symmetry == "symmetric" ? static_cast<std::size_t>(2 * nnz) : static_cast<std::size_t>(nnz));

        for(Eigen::Index k = 0; k < nnz; ++k) {
            if(!std::getline(file, line)) throw std::runtime_error(std::format("Unexpected end of Matrix Market data in: {}", path.string()));
            if(line.empty() || line.front() == '%') {
                --k;
                continue;
            }

            std::istringstream entry_stream(line);
            Eigen::Index       i     = 0;
            Eigen::Index       j     = 0;
            Scalar             value = 1.0;
            entry_stream >> i >> j;
            if(field != "pattern") entry_stream >> value;
            if(!entry_stream || i <= 0 || j <= 0 || i > rows || j > cols) {
                throw std::runtime_error(std::format("Invalid Matrix Market entry in: {}", path.string()));
            }

            --i;
            --j;
            triplets.emplace_back(i, j, value);
            if(symmetry == "symmetric" && i != j) triplets.emplace_back(j, i, value);
        }

        SparseMatrix matrix(rows, cols);
        matrix.setFromTriplets(triplets.begin(), triplets.end());
        matrix.makeCompressed();
        return matrix;
    }

    DenseMatrix seeded_initial_guess(Eigen::Index rows, Eigen::Index cols, unsigned int seed) {
        DenseMatrix                            guess(rows, cols);
        std::mt19937                           rng(seed);
        std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);
        for(Eigen::Index col = 0; col < cols; ++col)
            for(Eigen::Index row = 0; row < rows; ++row) guess(row, col) = dist(rng);
        return guess;
    }

    Eigen::Index solve_and_count_matvecs(const SparseMatrix &matrix, Eigen::Index ncv) {
        auto Aop = grit::matvec<Scalar>(matrix.rows(), [&](const DenseMatrixRef &X) -> DenseMatrix { return matrix * X; });

        grit::standard::gdplusk<Scalar> solver(Aop);
        solver.config.nev                           = 1;
        solver.config.ncv                           = ncv;
        solver.config.block_size                    = 1;
        solver.config.max_basis_blocks              = ncv;
        solver.config.ritz                          = grit::OptRitz::SR;
        solver.config.max_iters                     = -1;
        solver.config.inner_max_iters               = 2000;
        solver.config.tol                           = 1e-12;
        solver.config.inner_tol                     = 0.1;
        solver.config.residual_correction_type      = grit::ResidualCorrectionType::AUTO;
        solver.config.use_refined_rayleigh_ritz     = true;
        solver.config.use_adaptive_inner_tolerance  = false;
        solver.config.auto_sat_eigval_threshold     = 1e-3;
        solver.config.auto_sat_rnorm_threshold      = 1e-2;
        solver.config.auto_jd_start_rnorm_threshold = 1e-5;
        solver.config.auto_cheap_probe_interval     = 3;
        constexpr unsigned int initial_guess_seed = 0;
        solver.set_initial_guess(seeded_initial_guess(matrix.rows(), solver.config.block_size, initial_guess_seed));
        solver.run();

        auto view = grit::solver_view<Scalar>(solver);
        REQUIRE(view.stopReason() == grit::StopReason::converged);
        REQUIRE(view.rNorms().size() == 1);
        REQUIRE(view.rNorms()(0) < 1e-12);
        return view.num_matvecs_total();
    }
}

TEST_CASE("Andrews matvec count does not regress") {
    const auto matrix = read_matrix_market(std::filesystem::path{GRIT_TEST_DATA_DIR} / "Andrews.mtx");

    struct Case {
        Eigen::Index ncv;
        Eigen::Index matvecs;
    };

    const std::vector<Case> cases{
        {.ncv = 8, .matvecs = 855},
        {.ncv = 12, .matvecs = 812},
        {.ncv = 16, .matvecs = 885},
    };

    for(const auto &c : cases) {
        const auto actual  = solve_and_count_matvecs(matrix, c.ncv);
        const auto allowed = static_cast<Eigen::Index>(std::ceil(static_cast<double>(c.matvecs) * 1.05));
        INFO(std::format("ncv {} baseline {} actual {} allowed {}", c.ncv, c.matvecs, actual, allowed));
        REQUIRE(actual <= allowed);
    }
}

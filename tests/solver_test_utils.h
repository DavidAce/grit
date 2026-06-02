#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <grit/grit.h>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace grit_test {
    inline constexpr double nos4_min_eigenvalue = 5.3795283692696985e-04;
    inline constexpr double nos4_max_eigenvalue = 8.4913778378058813e-01;
    inline constexpr double nos4_condition      = 1.5784613919525875e+03;

    inline std::string lower_copy(std::string str) {
        std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return str;
    }

    inline std::filesystem::path data_dir() {
#ifdef GRIT_TEST_DATA_DIR
        return std::filesystem::path{GRIT_TEST_DATA_DIR};
#else
        return std::filesystem::path{"../../data"};
#endif
    }

    template<typename Scalar>
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> read_matrix_market_dense(const std::filesystem::path &path) {
        using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

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
        if(rows != cols) throw std::runtime_error(std::format("Solver tests require a square matrix: {}", path.string()));

        Matrix A = Matrix::Zero(rows, cols);
        for(Eigen::Index k = 0; k < nnz; ++k) {
            if(!std::getline(file, line)) throw std::runtime_error(std::format("Unexpected end of Matrix Market data in: {}", path.string()));
            if(line.empty() || line.front() == '%') {
                --k;
                continue;
            }

            std::istringstream entry_stream(line);
            Eigen::Index       i     = 0;
            Eigen::Index       j     = 0;
            Scalar             value = Scalar{1};
            entry_stream >> i >> j;
            if(field != "pattern") entry_stream >> value;
            if(!entry_stream || i <= 0 || j <= 0 || i > rows || j > cols) {
                throw std::runtime_error(std::format("Invalid Matrix Market entry in: {}", path.string()));
            }

            --i;
            --j;
            A(i, j) += value;
            if(symmetry == "symmetric" && i != j) A(j, i) += value;
        }
        return A;
    }

    template<typename Scalar>
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> nos4_matrix() {
        return read_matrix_market_dense<Scalar>(data_dir() / "nos4.mtx");
    }

    template<typename Scalar>
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> seeded_initial_guess(Eigen::Index rows, Eigen::Index cols, std::uint32_t seed) {
        using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

        Matrix                                 V(rows, cols);
        std::mt19937                           rng(seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for(Eigen::Index col = 0; col < cols; ++col)
            for(Eigen::Index row = 0; row < rows; ++row) V(row, col) = static_cast<Scalar>(dist(rng));
        return V;
    }

    template<typename VectorReal>
    std::vector<Eigen::Index> expected_ritz_indices(const VectorReal &evals, grit::Ritz ritz, Eigen::Index nev) {
        std::vector<Eigen::Index> idx(static_cast<std::size_t>(evals.size()));
        std::iota(idx.begin(), idx.end(), Eigen::Index{0});

        auto less = [&](Eigen::Index i, Eigen::Index j) {
            switch(ritz) {
                case grit::Ritz::LR: return evals(i) > evals(j);
                case grit::Ritz::LM: return std::abs(evals(i)) > std::abs(evals(j));
                case grit::Ritz::SR: return evals(i) < evals(j);
                case grit::Ritz::SM: return std::abs(evals(i)) < std::abs(evals(j));
                case grit::Ritz::NONE: return evals(i) < evals(j);
            }
            return evals(i) < evals(j);
        };

        std::partial_sort(idx.begin(), idx.begin() + nev, idx.end(), less);
        idx.resize(static_cast<std::size_t>(nev));
        return idx;
    }

    template<typename VectorReal>
    VectorReal expected_ritz_values(const VectorReal &evals, grit::Ritz ritz, Eigen::Index nev) {
        auto idx = expected_ritz_indices(evals, ritz, nev);

        VectorReal expected(nev);
        for(Eigen::Index i = 0; i < nev; ++i) expected(i) = evals(idx[static_cast<std::size_t>(i)]);
        return expected;
    }
}

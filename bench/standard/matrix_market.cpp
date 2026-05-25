#include "matrix_market.h"

#include "text.h"

#include <Eigen/SparseCore>
#include <format>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace bench_standard {
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
}

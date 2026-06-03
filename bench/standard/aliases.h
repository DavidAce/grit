#pragma once

#include <Eigen/SparseCore>
#include <grit/grit.h>
#include <string_view>
#include <variant>

namespace bench_standard {
    enum class Algo {
        gdplusk,
        lanczos,
        lobpcg,
    };

    constexpr std::string_view algo_name(Algo algo) {
        switch(algo) {
            case Algo::gdplusk: return "gdplusk";
            case Algo::lanczos: return "lanczos";
            case Algo::lobpcg: return "lobpcg";
        }
        return "gdplusk";
    }

    using Scalar             = double;
    using DenseMatrix        = grit::Matvec<Scalar>::MatrixType;
    using DenseMatrixRef     = Eigen::Ref<const DenseMatrix>;
    using SparseMatrix       = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;
    using ResidualCorrection = grit::ResidualCorrectionType;
}

#pragma once

#include <Eigen/SparseCore>
#include <grit/grit.h>

namespace bench_standard {
    using Scalar             = double;
    using DenseMatrix        = grit::Matvec<Scalar>::MatrixType;
    using DenseMatrixRef     = Eigen::Ref<const DenseMatrix>;
    using SparseMatrix       = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;
    using Solver             = grit::standard::gdplusk<Scalar>;
    using ResidualCorrection = grit::ResidualCorrectionType;
}

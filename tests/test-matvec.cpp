#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <grit/grit.h>

#include <cmath>
#include <optional>

namespace {
    template<typename VecA, typename VecB>
    void require_close(const VecA &a, const VecB &b, double tol) {
        REQUIRE(a.size() == b.size());
        for(Eigen::Index i = 0; i < a.size(); ++i) REQUIRE(std::abs(a(i) - b(i)) < tol);
    }
}

TEST_CASE("matvec raw callback applies operator") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(3, 3);
    A_matrix << 1.0, 0.1, 0.0,
        0.1, 2.0, 0.2,
        0.0, 0.2, 3.0;

    auto A = grit::matvec<double>(A_matrix.rows(), grit::raw, [&](const double *X, double *Y, Eigen::Index rows, Eigen::Index cols) {
        Eigen::Map<const Matrix> X_map(X, rows, cols);
        Eigen::Map<Matrix>       Y_map(Y, rows, cols);
        Y_map.noalias() = A_matrix * X_map;
    });

    Matrix X = Matrix::Identity(3, 2);
    Matrix Y = A.MultAX(X);
    require_close(Y.col(0), (A_matrix * X).col(0), 1e-12);
    require_close(Y.col(1), (A_matrix * X).col(1), 1e-12);
}

TEST_CASE("preconditioner callbacks are invoked") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(3, 3);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    int calc_count = 0;
    int prec_count = 0;
    A.set_CalcPc([&](double) { calc_count++; });
    A.set_MultPX([&](const Eigen::Ref<const Matrix> &X, const Eigen::Ref<const grit::MatVec<double>::VectorReal> &,
                     std::optional<const Eigen::Ref<const Matrix>>) {
        prec_count += static_cast<int>(X.cols());
        return X;
    });

    A.CalcPc(0.0);
    auto evals = grit::MatVec<double>::VectorReal::Zero(1);
    (void) A.MultPX(Matrix::Identity(3, 1), evals);

    REQUIRE(calc_count == 1);
    REQUIRE(prec_count == 1);
}

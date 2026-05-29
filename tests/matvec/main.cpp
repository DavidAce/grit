#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <grit/grit.h>

#include <cmath>

namespace {
    template<typename VecA, typename VecB>
    void require_close(const VecA &a, const VecB &b, double tol) {
        REQUIRE(a.size() == b.size());
        for(Eigen::Index i = 0; i < a.size(); ++i) REQUIRE(std::abs(a(i) - b(i)) < tol);
    }
}

TEST_CASE("matvec ptr callback applies operator") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(3, 3);
    A_matrix << 1.0, 0.1, 0.0,
        0.1, 2.0, 0.2,
        0.0, 0.2, 3.0;

    auto A = grit::matvec<double>(A_matrix.rows(), grit::ptr, [&](const double *X, double *Y, Eigen::Index rows, Eigen::Index cols) {
        Eigen::Map<const Matrix> X_map(X, rows, cols);
        Eigen::Map<Matrix>       Y_map(Y, rows, cols);
        Y_map.noalias() = A_matrix * X_map;
    });

    Matrix X = Matrix::Identity(3, 2);
    Matrix Y = A.mult(X);
    require_close(Y.col(0), (A_matrix * X).col(0), 1e-12);
    require_close(Y.col(1), (A_matrix * X).col(1), 1e-12);
}

TEST_CASE("preconditioner callbacks are invoked") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = Matrix::Identity(3, 3);
    auto   A        = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    int calc_count = 0;
    int prec_count = 0;
    A.set_preconditioner_update([&](double) { calc_count++; });
    A.set_preconditioner_apply([&](const Eigen::Ref<const grit::Matvec<double>::VectorType> &x, Eigen::Ref<grit::Matvec<double>::VectorType> y, double) {
        prec_count++;
        y = x;
    });

    A.preconditioner_update(0.0);
    grit::Matvec<double>::VectorType x = grit::Matvec<double>::VectorType::Ones(3);
    grit::Matvec<double>::VectorType y = grit::Matvec<double>::VectorType::Zero(3);
    A.preconditioner_apply(x, y, 0.0);

    REQUIRE(calc_count == 1);
    REQUIRE(prec_count == 1);
    require_close(x, y, 1e-12);
}

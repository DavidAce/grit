#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include <cmath>
#include <grit/grit.h>

namespace {
    template<typename VecA, typename VecB> void require_close(const VecA &a, const VecB &b, double abstol) {
        REQUIRE(a.size() == b.size());
        for(Eigen::Index i = 0; i < a.size(); ++i) REQUIRE(std::abs(a(i) - b(i)) < abstol);
    }
}

TEST_CASE("matvec ptr callback applies operator") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(3, 3);
    A_matrix << 1.0, 0.1, 0.0, 0.1, 2.0, 0.2, 0.0, 0.2, 3.0;

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
    REQUIRE(A.num_pc_update == 1);
    REQUIRE(A.num_pc == 1);
    REQUIRE(A.t_precond_update->get_tic_count() == 1);
    REQUIRE(A.t_precond->get_tic_count() == 1);
    require_close(x, y, 1e-12);
}

TEST_CASE("matvec reset clears counters and timers") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    auto A = grit::matvec<double>(3, [](const auto &X) { return Matrix(X); });
    A.set_preconditioner_update([](double) {});
    A.set_preconditioner_apply([](const auto &x, auto y, double) { y = x; });

    Matrix X = Matrix::Identity(3, 1);
    A.mult(X);
    A.preconditioner_update(0.0);
    grit::Matvec<double>::VectorType y(3);
    A.preconditioner_apply(X.col(0), y, 0.0);
    A.reset();

    REQUIRE(A.num_mv == 0);
    REQUIRE(A.num_pc == 0);
    REQUIRE(A.num_pc_update == 0);
    REQUIRE(A.t_mult->get_tic_count() == 0);
    REQUIRE(A.t_precond->get_tic_count() == 0);
    REQUIRE(A.t_precond_update->get_tic_count() == 0);
    REQUIRE(A.t_mult->get_time() == Approx(0.0));
    REQUIRE(A.t_precond->get_time() == Approx(0.0));
    REQUIRE(A.t_precond_update->get_time() == Approx(0.0));
}

TEST_CASE("timer accumulated laps follow measured intervals") {
    grit::tid::ur timer;

    timer += 0.25;
    REQUIRE(timer.get_time() == Approx(0.25));
    REQUIRE(timer.get_last_interval() == Approx(0.25));
    REQUIRE(timer.get_tic_count() == 1);
    REQUIRE(timer.get_time_lap() == Approx(0.25));
    REQUIRE(timer.restart_time_lap() == Approx(0.25));
    REQUIRE(timer.get_last_time_lap() == Approx(0.25));
    REQUIRE(timer.get_time_lap() == Approx(0.0));

    timer += 0.5;
    REQUIRE(timer.get_time() == Approx(0.75));
    REQUIRE(timer.get_last_interval() == Approx(0.5));
    REQUIRE(timer.get_tic_count() == 2);
    REQUIRE(timer.get_time_lap() == Approx(0.5));
    REQUIRE(timer.restart_time_lap() == Approx(0.5));
    REQUIRE(timer.get_last_time_lap() == Approx(0.5));
}

int main(int argc, char **argv) {
    Catch::Session session;
    const int      return_code = session.applyCommandLine(argc, argv);
    if(return_code != 0) return return_code;
    return session.run();
}

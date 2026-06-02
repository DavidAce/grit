#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include "solver_test_utils.h"
#include <cmath>
#include <Eigen/Eigenvalues>
#include <format>
#include <grit/grit.h>
#include <string>
#include <string_view>
#include <unistd.h>

namespace {
    int test_log_fd = STDERR_FILENO;

    void write_test_log(const std::string &message) {
        const char *data = message.data();
        auto        size = message.size();
        while(size != 0) {
            const auto written = ::write(test_log_fd, data, size);
            if(written <= 0) return;
            data += written;
            size -= static_cast<std::size_t>(written);
        }
    }

    template<typename VecA, typename VecB>
    void require_close(const VecA &a, const VecB &b, double tol) {
        REQUIRE(a.size() == b.size());
        for(Eigen::Index i = 0; i < a.size(); ++i) REQUIRE(std::abs(a(i) - b(i)) < tol);
    }

    template<typename VecA, typename VecB>
    void print_eigenvalue_comparison(std::string_view label, const VecA &computed, const VecB &exact, Eigen::Index count) {
        write_test_log(std::format("{} eigenvalue comparison:\n", label));
        for(Eigen::Index i = 0; i < count; ++i) {
            const auto diff = std::abs(computed(i) - exact(i));
            write_test_log(std::format("  [{}] computed {:.16e} exact {:.16e} abs_diff {:.3e}\n", i, computed(i), exact(i), diff));
        }
    }
}

TEST_CASE("generalized lanczos matches dense eigensolver") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.0, 0.0, 0.5, 2.0, 0.25, 0.0, 0.0, 0.0, 0.25, 5.0, 0.5, 0.0, 0.0, 0.0, 0.5, 6.0;

    Matrix B_matrix = Matrix::Identity(5, 5);
    B_matrix.diagonal() << 1.0, 1.5, 2.0, 2.5, 3.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    Matrix V = Matrix::Identity(A_matrix.rows(), A_matrix.rows());

    grit::generalized::lanczos<double> solver(A, B);
    solver.config.nev        = 1;
    solver.config.ncv        = A_matrix.rows();
    solver.config.block_size = 1;
    solver.config.ritz       = grit::OptRitz::SR;
    solver.config.max_iters  = 20;
    solver.set_initial_guess(V);
    solver.run();

    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> exact(A_matrix, B_matrix);
    auto                                             view = grit::solver_view<double>(solver);
    REQUIRE(view.stopReason() == grit::StopReason::converged);
    print_eigenvalue_comparison("generalized lanczos", view.eigVal(), exact.eigenvalues(), view.eigVal().size());
    require_close(view.eigVal(), exact.eigenvalues().head(1), 1e-10);
}

TEST_CASE("generalized lanczos handles nos4 restart block search") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = grit_test::nos4_matrix<double>();
    Matrix B_matrix = Matrix::Identity(A_matrix.rows(), A_matrix.cols());

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    grit::generalized::lanczos<double> solver(A, B);
    solver.config.nev                 = 2;
    solver.config.ncv                 = 20;
    solver.config.block_size          = 2;
    solver.config.ritz                = grit::OptRitz::SR;
    solver.config.max_iters           = 80;
    solver.config.tol                 = 1e-9;
    solver.config.use_b_inner_product = false;
    solver.set_initial_guess(grit_test::seeded_initial_guess<double>(A_matrix.rows(), solver.config.block_size, 71));
    solver.run();

    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> exact(A_matrix, B_matrix);
    auto                                             expected = grit_test::expected_ritz_values(exact.eigenvalues(), solver.config.ritz, solver.config.nev);
    auto                                             view     = grit::solver_view<double>(solver);
    REQUIRE_FALSE(grit::has_flag(view.stopReason(), grit::StopReason::invalid_input));
    REQUIRE(std::abs(exact.eigenvalues()(0) - grit_test::nos4_min_eigenvalue) < 1e-12);
    REQUIRE(std::abs(exact.eigenvalues()(exact.eigenvalues().size() - 1) - grit_test::nos4_max_eigenvalue) < 1e-12);
    REQUIRE(std::abs(exact.eigenvalues()(exact.eigenvalues().size() - 1) / exact.eigenvalues()(0) - grit_test::nos4_condition) < 1e-8);
    print_eigenvalue_comparison("generalized lanczos nos4 restart", view.eigVal(), expected, view.eigVal().size());
    require_close(view.eigVal(), expected, 1e-7);
}

TEST_CASE("generalized lanczos supports all Ritz targets on nos4") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = grit_test::nos4_matrix<double>();
    Matrix B_matrix = Matrix::Identity(A_matrix.rows(), A_matrix.cols());

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> exact(A_matrix, B_matrix);
    for(auto ritz : {grit::OptRitz::SR, grit::OptRitz::LR, grit::OptRitz::SM, grit::OptRitz::LM}) {
        grit::generalized::lanczos<double> solver(A, B);
        solver.config.nev                 = 2;
        solver.config.ncv                 = A_matrix.rows();
        solver.config.block_size          = 2;
        solver.config.ritz                = ritz;
        solver.config.max_iters           = 100;
        solver.config.tol                 = 1e-9;
        solver.config.use_b_inner_product = true;
        solver.set_initial_guess(grit_test::seeded_initial_guess<double>(A_matrix.rows(), solver.config.block_size, 80 + static_cast<int>(ritz)));
        solver.run();

        auto expected = grit_test::expected_ritz_values(exact.eigenvalues(), ritz, solver.config.nev);
        auto view     = grit::solver_view<double>(solver);
        REQUIRE_FALSE(grit::has_flag(view.stopReason(), grit::StopReason::invalid_input));
        print_eigenvalue_comparison(std::format("generalized lanczos {}", grit::enum2sv(ritz)), view.eigVal(), expected, view.eigVal().size());
        require_close(view.eigVal(), expected, 1e-7);
    }
}

TEST_CASE("generalized lanczos rejects nev larger than block_size") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix = grit_test::nos4_matrix<double>();
    Matrix B_matrix = Matrix::Identity(A_matrix.rows(), A_matrix.cols());

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    grit::generalized::lanczos<double> solver(A, B);
    solver.config.nev        = 3;
    solver.config.block_size = 2;
    solver.config.ncv        = 20;

    REQUIRE_THROWS_WITH(solver.run(), Catch::Matchers::Contains("lanczos config error: nev must not exceed block_size"));
}

TEST_CASE("generalized lanczos with B as A squared targets A smallest magnitude through LM") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Vector<double, Eigen::Dynamic>;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.0, 0.0, 0.5, 2.0, 0.25, 0.0, 0.0, 0.0, 0.25, 5.0, 0.5, 0.0, 0.0, 0.0, 0.5, 6.0;

    Matrix B_matrix = A_matrix * A_matrix;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    Matrix V = Matrix::Identity(A_matrix.rows(), A_matrix.rows());

    grit::generalized::lanczos<double> solver(A, B);
    solver.config.nev        = 1;
    solver.config.ncv        = A_matrix.rows();
    solver.config.block_size = 1;
    solver.config.ritz       = grit::OptRitz::LM;
    solver.config.max_iters  = 20;
    solver.set_initial_guess(V);
    solver.run();

    Eigen::SelfAdjointEigenSolver<Matrix> exact_A(A_matrix);
    Vector                                expected(1);
    expected << 1.0 / exact_A.eigenvalues()(0);

    auto view = grit::solver_view<double>(solver);
    REQUIRE(view.stopReason() == grit::StopReason::converged);
    print_eigenvalue_comparison("generalized lanczos B=A^2 LM", view.eigVal(), expected, view.eigVal().size());
    write_test_log(std::format("  recovered A eigenvalue {:.16e} exact SM {:.16e} abs_diff {:.3e}\n", 1.0 / view.eigVal()(0), exact_A.eigenvalues()(0),
                               std::abs(1.0 / view.eigVal()(0) - exact_A.eigenvalues()(0))));
    require_close(view.eigVal(), expected, 1e-10);
    REQUIRE(std::abs(1.0 / view.eigVal()(0) - exact_A.eigenvalues()(0)) < 1e-10);
}

int main(int argc, char **argv) {
    test_log_fd = ::dup(STDERR_FILENO);
    Catch::Session session;
    const int      return_code = session.applyCommandLine(argc, argv);
    if(return_code != 0) return return_code;
    const int result = session.run();
    if(test_log_fd >= 0) ::close(test_log_fd);
    return result;
}

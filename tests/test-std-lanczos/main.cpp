#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include <grit/grit.h>

#include <Eigen/Eigenvalues>
#include <cmath>
#include <format>
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

TEST_CASE("standard lanczos matches dense eigensolver") {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(5, 5);
    A_matrix << 4.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 3.0, 0.5, 0.0, 0.0,
        0.0, 0.5, 2.0, 0.25, 0.0,
        0.0, 0.0, 0.25, 5.0, 0.5,
        0.0, 0.0, 0.0, 0.5, 6.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    Matrix V = Matrix::Identity(A_matrix.rows(), A_matrix.rows());

    grit::standard::lanczos<double> solver(A);
    solver.config.nev              = 2;
    solver.config.ncv              = A_matrix.rows();
    solver.config.block_size       = 1;
    solver.config.max_basis_blocks = A_matrix.rows();
    solver.config.ritz             = grit::OptRitz::SR;
    solver.config.max_iters        = 20;
    solver.set_initial_guess(V);
    solver.run();

    Eigen::SelfAdjointEigenSolver<Matrix> exact(A_matrix);
    auto view = grit::solver_view<double>(solver);
    REQUIRE(view.stopReason() == grit::StopReason::converged);
    print_eigenvalue_comparison("standard lanczos", view.eigVal(), exact.eigenvalues(), view.eigVal().size());
    require_close(view.eigVal(), exact.eigenvalues().head(2), 1e-10);
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

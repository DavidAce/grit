#include <grit/grit.h>

#include <Eigen/Core>
#include <print>

int main() {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Matrix A_matrix(4, 4);
    A_matrix << 1.0, 0.1, 0.0, 0.0,
        0.1, 2.0, 0.2, 0.0,
        0.0, 0.2, 3.0, 0.3,
        0.0, 0.0, 0.3, 4.0;

    Matrix B_matrix = A_matrix * A_matrix;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });
    auto B = grit::matvec<double>(B_matrix.rows(), [&](auto const &X) { return B_matrix * X; });

    grit::generalized::lanczos<double> solver(A, B);
    solver.config.nev              = 1;
    solver.config.ncv              = A_matrix.rows();
    solver.config.block_size       = 1;
    solver.config.ritz             = grit::OptRitz::LM;
    solver.set_initial_guess(Matrix::Identity(A_matrix.rows(), A_matrix.rows()));
    solver.run();

    auto view = grit::solver_view<double>(solver);
    std::println("generalized lambda = {:.16e}", view.eigVal()(0));
    std::println("recovered A eigenvalue = {:.16e}", 1.0 / view.eigVal()(0));
}

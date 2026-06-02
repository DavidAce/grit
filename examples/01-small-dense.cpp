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

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    grit::standard::gdplusk<double> solver(A);
    solver.config.nev              = 1;
    solver.config.ncv              = A_matrix.rows();
    solver.config.block_size       = 1;
    solver.config.ritz             = grit::Ritz::SR;
    solver.set_initial_guess(Matrix::Identity(A_matrix.rows(), A_matrix.rows()));
    solver.run();

    auto view = grit::solver_view<double>(solver);
    for(Eigen::Index i = 0; i < view.eigVal().size(); ++i) {
        std::println("lambda[{}] = {:.16e}", i, view.eigVal()(i));
    }
}

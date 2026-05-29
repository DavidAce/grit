#include <grit/grit.h>

#include <Eigen/Core>
#include <iostream>

int main() {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    Matrix A_matrix(4, 4);
    A_matrix << 1.0, 0.1, 0.0, 0.0,
        0.1, 2.0, 0.2, 0.0,
        0.0, 0.2, 3.0, 0.3,
        0.0, 0.0, 0.3, 4.0;

    auto A = grit::matvec<double>(A_matrix.rows(), [&](auto const &X) { return A_matrix * X; });

    Matrix V = Matrix::Random(A_matrix.rows(), 2);
    grit::standard::gdplusk<double> solver(A);
    solver.config.nev              = 2;
    solver.config.ncv              = A_matrix.rows();
    solver.config.block_size       = 1;
    solver.config.max_basis_blocks = A_matrix.rows();
    solver.config.ritz             = grit::OptRitz::SR;
    solver.set_initial_guess(V);
    solver.run();

    auto view = grit::solver_view<double>(solver);
    std::cout << view.eigVal().transpose() << '\n';
}

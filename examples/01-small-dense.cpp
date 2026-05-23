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
    grit::standard::problem<double> problem(A);
    grit::gdplusk_config<double>    cfg;
    cfg.nev = 2;
    cfg.ncv = A_matrix.rows();
    cfg.ritz = grit::OptRitz::SR;
    cfg.set_initial_guess(V);
    grit::standard::gdplusk<double> solver(problem, cfg);
    solver.run();

    std::cout << solver.result().eigVal().transpose() << '\n';
}

#include <grit/grit.h>

int main() {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    Matrix A_matrix(2, 2);
    A_matrix << 1.0, 0.0, 0.0, 2.0;
    auto A = grit::matvec<double>(2, [&](auto const &X) { return A_matrix * X; });
    Matrix V = Matrix::Random(2, 1);
    grit::standard::gdplusk<double> solver(A);
    solver.config.nev              = 1;
    solver.config.ncv              = 2;
    solver.config.block_size       = 1;
    solver.config.max_basis_blocks = 2;
    solver.config.ritz             = grit::OptRitz::SR;
    solver.set_initial_guess(V);
    solver.run();
    return grit::solver_view<double>(solver).stopReason() == grit::StopReason::converged ? 0 : 1;
}

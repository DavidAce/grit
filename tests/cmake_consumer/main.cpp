#include <grit/grit.h>

int main() {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    Matrix A_matrix(2, 2);
    A_matrix << 1.0, 0.0, 0.0, 2.0;
    auto A = grit::matvec<double>(2, [&](auto const &X) { return A_matrix * X; });
    Matrix V = Matrix::Random(2, 1);
    grit::standard::problem<double> problem(A);
    grit::gdplusk_config<double>    cfg;
    cfg.nev = 1;
    cfg.ncv = 2;
    cfg.ritz = grit::OptRitz::SR;
    cfg.set_initial_guess(V);
    grit::standard::gdplusk<double> solver(problem, cfg);
    solver.run();
    return solver.result().stopReason() == grit::StopReason::converged ? 0 : 1;
}

#include <iostream>

#include "sdqp/sdqp.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char **argv)
{
    int m = 5;
    Eigen::Matrix<double, 3, 3> Q, Q_prox;
    Eigen::Matrix<double, 3, 1> c, c_prox;
    Eigen::Matrix<double, 3, 1> x, x_hat;        // decision variables
    Eigen::Matrix<double, -1, 3> A(m, 3); // constraint matrix
    Eigen::VectorXd b(m);                 // constraint bound

    Q << 8.0, -6.0, 2.0, -6.0, 6.0, -3.0, 2.0, -3.0, 2.0;

    c << 1.0, 3.0, -2.0;

    A << 0.0, -1.0, -2.0,
        -1.0, 1.0, -3.0,
        1.0, -2.0, 0.0,
        -1.0, -2.0, -1.0,
        3.0, 5.0, 1.0;

    b << -1.0, 2.0, 7.0, 2.0, -1.0;

    x_hat << 0.0, -1.0, 2.0; // a feasible initial guess

    // x = x_hat;

    constexpr double eps = 1e-5;
    constexpr int max_iterations = 50;
    Eigen::MatrixXd Identity_3 = Eigen::MatrixXd::Identity(3, 3);

    double rho = 1.0, tol = 1;
    int iteration = 0;

    while (tol > eps && iteration <= max_iterations) {
        ++iteration;
        Q_prox = Q + 1.0 / rho * Identity_3;
        c_prox = c -  1.0 / rho * x_hat;
        double minobj = sdqp::sdqp<3>(Q_prox, c_prox, A, b, x);
        std::cout << "=================================" << std::endl;
        std::cout << "iteration: " << iteration << endl;
        std::cout << "optimal sol: " << x.transpose() << std::endl;
        std::cout << "optimal obj: " << minobj << std::endl;
        std::cout << "cons precision: " << (A * x - b).maxCoeff() << std::endl;
        tol = (x - x_hat).norm() / std::max(1.0, x.norm());
        rho = std::min(rho * 10, 1e6);
        x_hat = x;
    }

    return 0;
}

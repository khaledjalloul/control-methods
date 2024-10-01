#include "state_space.hpp"

LTI_StateSpaceModel::LTI_StateSpaceModel(Matrix A, Matrix B) : A_(A), B_(B)
{
    nx_ = A.rows();
    nu_ = B.cols();

    // int num_inputs = sizeof(possible_inputs) / sizeof(*possible_inputs);

    // for (int i = 0; i < num_inputs; i++)
    // {
    //     possible_inputs[i] = -2 + i * 4.0 / num_inputs;
    // }
}

Matrix LTI_StateSpaceModel::x_next(Vector x, Vector u)
{
    return A_ * x + B_ * u;
}

Vector LTI_StateSpaceModel::find_steady_state(Vector desired_x_ss, std::optional<Vector> desired_u_ss)
{
    OsqpEigen::Solver solver;

    int n = desired_x_ss.size() + (desired_u_ss.has_value() ? desired_u_ss.value().size() : 0);

    Eigen::SparseMatrix<double> H(n, n);
    H.setIdentity();

    Vector g = Vector(n);
    g.topRows(nx_) << -desired_x_ss;
    if (desired_u_ss.has_value())
        g.bottomRows(nu_) << -desired_u_ss.value();

    Eigen::SparseMatrix<double> C(nx_, nx_ + nu_);
    C.setZero();
    C.block(0, 0, nx_, nx_) = Matrix::Identity(nx_, nx_) - A_;
    C.block(0, nx_, nx_, nu_) = -B_;
    // if (desired_u_ss.has_value())
    //     C.block(nx_, 0, nu_, nu_) = Matrix::Identity(nu_, nu_);

    // Solve (x - x_ss)^T * Q * (x - x_ss) + (u - u_ss)^T * R * (u - u_ss)
    // subject to x = A * x + B * u ->
    // [x, u] = [A, B, 0, I] * [x, u]

    Vector l_u = Vector::Zero(n);

    solver.data()->setNumberOfVariables(n);
    solver.data()->setNumberOfConstraints(nx_);

    solver.data()->setHessianMatrix(H);
    solver.data()->setGradient(g);
    solver.data()->setLinearConstraintsMatrix(C);
    solver.data()->setBounds(l_u, l_u);

    solver.initSolver();
    solver.solveProblem();

    Vector sol = solver.getSolution();
    return sol;
}

// // float LTI_StateSpaceModel::get_reward(Vector x, Vector u, Vector x_ss, Vector u_ss)
// {
//     return (x - xss).dot(x - xss) + pow(u - uss, 2) * 0.1;
// }

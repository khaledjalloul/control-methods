#include "state_space.hpp"

LTIStateSpaceModel::LTIStateSpaceModel(Matrix A, Matrix B) : A(A), B(B)
{
    nx = A.rows();
    nu = B.cols();

    ny = nx;
    C = Matrix::Identity(nx, nx);

    solver_.settings()->setVerbosity(false);

    // int num_inputs = sizeof(possible_inputs) / sizeof(*possible_inputs);

    // for (int i = 0; i < num_inputs; i++)
    // {
    //     possible_inputs[i] = -2 + i * 4.0 / num_inputs;
    // }
}

Matrix LTIStateSpaceModel::x_next(Vector x, Vector u)
{
    return A * x + B * u;
}

Vector LTIStateSpaceModel::y(Vector x)
{
    return C * x;
}

Matrix LTIStateSpaceModel::y_mat(Matrix x_mat)
{
    return C * x_mat;
}

SteadyState LTIStateSpaceModel::find_steady_state(Vector desired_x_ss, std::optional<Vector> desired_u_ss)
{
    int n = nx + nu;
    solver_.data()->setNumberOfVariables(n);

    Matrix H = Matrix::Zero(n, n);
    H.block(0, 0, nx, nx) = Matrix::Identity(nx, nx);
    if (desired_u_ss.has_value())
        H.block(nx, nx, nu, nu) = Matrix::Identity(nu, nu) * 0.3;
    solver_.data()->setHessianMatrix((Eigen::SparseMatrix<double>)H.sparseView());

    Vector g = Vector::Zero(n);
    g.head(nx) << -desired_x_ss;
    if (desired_u_ss.has_value())
        g.segment(nx, nu) << -desired_u_ss.value();
    solver_.data()->setGradient(g);

    Matrix C = Matrix::Zero(nx, nx + nu);
    C.block(0, 0, nx, nx) = Matrix::Identity(nx, nx) - A;
    C.block(0, nx, nx, nu) = -B;
    solver_.data()->setNumberOfConstraints(nx);
    solver_.data()->setLinearConstraintsMatrix((Eigen::SparseMatrix<double>)C.sparseView());

    Vector l_u = Vector::Zero(nx);
    solver_.data()->setBounds(l_u, l_u);

    solver_.initSolver();
    solver_.solveProblem();
    Vector sol = solver_.getSolution();

    return {sol.head(nx), sol.tail(nu)};
}

// // float LTIStateSpaceModel::get_reward(Vector x, Vector u, Vector x_ss, Vector u_ss)
// {
//     return (x - xss).dot(x - xss) + pow(u - uss, 2) * 0.1;
// }

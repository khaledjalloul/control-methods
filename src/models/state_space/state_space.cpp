#include "state_space.hpp"

LTIStateSpaceModel::LTIStateSpaceModel(Matrix A, Matrix B) : A(A), B(B)
{
    nx = A.rows();
    nu = B.cols();

    ny = nx;
    C = Matrix::Identity(nx, nx);

    solver_.settings()->setVerbosity(false);
}

Matrix LTIStateSpaceModel::x_next(Vector x, Vector u)
{
    return A * x + B * u;
}

SteadyState LTIStateSpaceModel::find_steady_state(Vector desired_x_ss, std::optional<Vector> desired_u_ss)
{
    solver_.clearSolver();
    solver_.data()->clearHessianMatrix();
    solver_.data()->clearLinearConstraintsMatrix();

    int n = nx + nu;
    solver_.data()->setNumberOfVariables(n);
    solver_.data()->setNumberOfConstraints(nx);

    Matrix H = Matrix::Zero(n, n);
    H.block(0, 0, nx, nx) = Matrix::Identity(nx, nx);
    if (desired_u_ss.has_value())
        H.block(nx, nx, nu, nu) = Matrix::Identity(nu, nu) * 0.3;

    Vector g = Vector::Zero(n);
    g.head(nx) << -desired_x_ss;
    if (desired_u_ss.has_value())
        g.segment(nx, nu) << -desired_u_ss.value();

    Matrix C = Matrix::Zero(nx, nx + nu);
    C.block(0, 0, nx, nx) = Matrix::Identity(nx, nx) - A;
    C.block(0, nx, nx, nu) = -B;

    Vector l_u = Vector::Zero(nx);

    solver_.data()->setHessianMatrix((Eigen::SparseMatrix<double>)H.sparseView());
    solver_.data()->setGradient(g);
    solver_.data()->setLinearConstraintsMatrix((Eigen::SparseMatrix<double>)C.sparseView());
    solver_.data()->setBounds(l_u, l_u);

    solver_.initSolver();
    solver_.solveProblem();

    Vector sol = solver_.getSolution();

    return {sol.head(nx), sol.tail(nu)};
}
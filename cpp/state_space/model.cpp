#include <iostream>
#include "model.h"

using namespace std;
using namespace Eigen;

LinearStateSpaceModel::LinearStateSpaceModel(MatrixXd A, MatrixXd B, MatrixXd C) : A(A), B(B)
{
    nx = A.rows();
    nu = B.cols();

    if (C == MatrixXd::Zero(0, 0))
        C = MatrixXd(nx, nx);
}

MatrixXd LinearStateSpaceModel::x_next(VectorXd x, VectorXd u)
{
    return A * x + B * u;
}

MatrixXd LinearStateSpaceModel::y(VectorXd x)
{
    return C * x;
}

void LinearStateSpaceModel::find_steady_state(VectorXd desired_x_ss, VectorXd desired_u_ss)
{
}

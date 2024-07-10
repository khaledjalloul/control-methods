#include "model.h"

LinearStateSpaceModel::LinearStateSpaceModel(MatrixXd A, MatrixXd B) : A(A), B(B)
{
    nx = A.rows();
    nu = B.cols();

    int numInputs = sizeof(possibleInputs) / sizeof(*possibleInputs);

    for (int i = 0; i < numInputs; i++)
    {
        possibleInputs[i] = -2 + i * 4.0 / numInputs;
    }
}

MatrixXd LinearStateSpaceModel::xNext(X x, U u)
{
    return A * x + B * u;
}

void LinearStateSpaceModel::findSteadyState(X desired_x_ss, U desired_u_ss)
{
}

float LinearStateSpaceModel::getReward(X x, U u, X xss, U uss)
{
    return (x - xss).dot(x - xss) + pow(u - uss, 2) * 0.1;
}

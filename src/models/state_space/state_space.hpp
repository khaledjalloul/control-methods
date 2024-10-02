#pragma once

#include <iostream>
#include <optional>
#include <eigen3/Eigen/Dense>
#include <OsqpEigen/OsqpEigen.h>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

struct SteadyState
{
    Vector x_ss;
    Vector u_ss;
};

class LTIStateSpaceModel
{
public:
    int nx, nu;
    Matrix A, B;

    LTIStateSpaceModel(Matrix A, Matrix B);

    Matrix x_next(Vector x, Vector u);

    SteadyState find_steady_state(Vector desired_x_ss, std::optional<Vector> desired_u_ss = std::nullopt);

    // float possibleInputs[20];
    // float get_reward(Vector x, Vector u, State x_ss, Vector u_ss);

private:
    OsqpEigen::Solver solver_;
};
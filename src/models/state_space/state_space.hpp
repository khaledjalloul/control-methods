#pragma once

#include <iostream>
#include <optional>
#include <eigen3/Eigen/Dense>
#include <OsqpEigen/OsqpEigen.h>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

class LTI_StateSpaceModel
{
public:
    LTI_StateSpaceModel(Matrix A, Matrix B);

    Matrix x_next(Vector x, Vector u);

    Vector find_steady_state(Vector desired_x_ss, std::optional<Vector> desired_u_ss = std::nullopt);

    // float possibleInputs[20];
    // float get_reward(Vector x, Vector u, State x_ss, Vector u_ss);

private:
    Matrix A_, B_, C_;
    int nx_, nu_;
};
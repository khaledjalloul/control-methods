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
    int nx, nu, ny;
    Matrix A, B, C;

    LTIStateSpaceModel(Matrix A, Matrix B);

    Matrix x_next(Vector x, Vector u);

    template <typename T>
    T y(T x);

    SteadyState find_steady_state(Vector desired_x_ss, std::optional<Vector> desired_u_ss = std::nullopt);

private:
    OsqpEigen::Solver solver_;
};

#include "state_space.tpp"
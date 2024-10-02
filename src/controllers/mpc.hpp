#pragma once

#include <iostream>
#include <memory>

#include <eigen3/Eigen/Dense>
#include <OsqpEigen/OsqpEigen.h>

#include "state_space.hpp"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

class MPC
{
public:
    MPC(const std::shared_ptr<LTIStateSpaceModel> &model, int K);

    Vector step(Vector x0, Vector x_ss, Vector u_ss);

private:
    int K_, nx_, nu_, num_vars_, num_constraints_;
    Matrix Q_, R_;
    double slack_cost_;
    std::shared_ptr<LTIStateSpaceModel> model_;
    OsqpEigen::Solver solver_;
};
#pragma once

#include <memory>

#include "state_space.hpp"

class MPC
{
public:
    MPC(const std::shared_ptr<LTIStateSpaceModel> &model, int K);

    Vector step(Vector x0, Vector x_ss, Vector u_ss);

private:
    int K_, nx_, nu_, num_vars_, num_constraints_;
    Matrix Q_, R_;
    std::shared_ptr<LTIStateSpaceModel> model_;
    OsqpEigen::Solver solver_;
};
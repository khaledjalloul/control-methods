#pragma once

#include <memory>

#include "state_space.hpp"

class DeePC
{
public:
    DeePC(const std::shared_ptr<LTIStateSpaceModel> &model);

    Vector step(Vector x0, Vector x_ss, Vector u_ss, Matrix u_past, Matrix y_past_, int current_index);

private:
    int ny_, nu_, num_vars_, num_constraints_, T_prev_, T_fut_, L_, num_hankel_columns_;
    Matrix Q_, R_, Hu_, Hy_;
    std::shared_ptr<LTIStateSpaceModel> model_;
    OsqpEigen::Solver solver_;
};
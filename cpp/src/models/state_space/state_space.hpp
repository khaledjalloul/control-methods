#pragma once

#include <iostream>
#include <eigen3/Eigen/Dense>

using Matrix = Eigen::MatrixXd;

template<typename State, typename Action>
class LTI_StateSpaceModel
{
private:
    Matrix A_, B_, C_;
    int nx_, nu_;

public:
    // float possibleInputs[20];

    LTI_StateSpaceModel(Matrix A, Matrix B);

    Matrix x_next(State x, Action u);

    void find_steady_state(State desired_x_ss, Action desired_u_ss = 0);

    // float get_reward(State x, Action u, State x_ss, Action u_ss);
};
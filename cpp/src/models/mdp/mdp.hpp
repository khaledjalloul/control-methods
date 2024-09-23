#pragma once

#include <iostream>
#include <tuple>
#include <eigen3/Eigen/Dense>

using Matrix = Eigen::MatrixXd;

template <typename State, typename Action>
class MarkovDecisionProcess
{
public:
    State x_initial;

    virtual int state_to_index(State x) = 0;

    virtual State index_to_state(int index) = 0;

    virtual std::tuple<int, Eigen::VectorXd> get_reward_and_trans_prob(int index, Action u) = 0;

    virtual bool is_done(State x) = 0;

    virtual State play(State x_old, Action u) = 0;
};
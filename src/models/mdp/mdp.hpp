#pragma once

#include <iostream>
#include <tuple>
#include <eigen3/Eigen/Dense>

using Matrix = Eigen::MatrixXd;

template <typename State, typename Action>
class MarkovDecisionProcess
{
public:
    virtual int state_to_index(State x) = 0;

    virtual State index_to_state(int x_index) = 0;

    virtual std::tuple<int, Eigen::VectorXd> get_reward_and_trans_prob(int x_index, Action u, State x_goal = 0) = 0;

    virtual bool is_done(State x, State x_goal = 0) = 0;

    virtual State play(State x_old, Action u) = 0;
};
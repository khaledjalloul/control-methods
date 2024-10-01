#pragma once

#include <iostream>
#include <tuple>
#include <eigen3/Eigen/Dense>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

struct RewardTransProb {
    int reward;
    Vector trans_prob;
};

template <typename State, typename Action>
class MarkovDecisionProcess
{
public:
    virtual int state_to_index(State x) = 0;

    virtual State index_to_state(int x_index) = 0;

    virtual RewardTransProb get_reward_and_trans_prob(int x_index, Action u, State x_goal = State{}) = 0;

    virtual bool is_done(State x, State x_goal = 0) = 0;

    virtual State play(State x_old, Action u) = 0;
};
#pragma once

#include <memory>
#include <iomanip>

#include "grid.hpp"
#include "tic_tac_toe.hpp"

template <class State, class Action>
class QLearning
{
    using MDP_Ptr = std::shared_ptr<MarkovDecisionProcess<State, Action>>;

public:
    QLearning(MDP_Ptr mdp, int nx, int nu, State x_initial, State x_goal);

    Matrix get_Q();

    Action sample_action(int index, int iter, int num_iters);

    Matrix get_policy();

    void train(int num_eps);

private:
    MDP_Ptr mdp_;
    int nx_, nu_, max_steps_per_episode_ = 100;
    float gamma_ = 0.9, epsilon_ = 0.9, RL_learning_rate_ = 0.5;
    Matrix Q_;
    State x_initial_, x_goal_;
};

#include "q_learning.tpp"
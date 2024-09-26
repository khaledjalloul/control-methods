#pragma once

#include <memory>

#include "../models/mdp/mdp.hpp"

template <class State, class Action>
class QLearning
{
private:
    std::shared_ptr<MarkovDecisionProcess<State, Action>> mdp_;
    int nx_, nu_, max_steps_per_episode_ = 100;
    float gamma_ = 0.9, epsilon_ = 0.9, RL_learning_rate_ = 0.5;
    Eigen::MatrixXd Q_;
    State x_initial_, x_goal_;

public:
    QLearning(std::shared_ptr<MarkovDecisionProcess<State, Action>> mdp, int nx, int nu, State x_initial, State x_goal);

    Eigen::MatrixXd get_Q();

    Action sample_action(int index, int iter, int num_iters);

    Eigen::MatrixXd get_policy();

    void train(int num_eps);
};
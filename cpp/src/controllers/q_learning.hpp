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

public:
    QLearning(std::shared_ptr<MarkovDecisionProcess<State, Action>> mdp, int nx, int nu);

    Action sample_action_from_Q(int index, Eigen::MatrixXd Q, int iter, int num_iters);

    Eigen::MatrixXd get_policy_from_Q(Eigen::MatrixXd Q);

    Eigen::MatrixXd RL_Q_learning(int num_eps);
};
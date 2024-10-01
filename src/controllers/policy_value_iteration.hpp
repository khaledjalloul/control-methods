#pragma once

#include <memory>

#include "grid.hpp"
#include "tic_tac_toe.hpp"

struct PolicyValue {
    Matrix policy;
    Vector V;
};

template <class State, class Action>
class PolicyValueIteration
{
    using MDP_Ptr = std::shared_ptr<MarkovDecisionProcess<State, Action>>;

public:
    PolicyValueIteration(MDP_Ptr mdp, int nx, int nu, State x_goal);

    Vector evaluate_policy(Matrix policy);

    PolicyValue improve_policy(Vector V);

    PolicyValue train_policy_iteration(int num_iters = 30);

    PolicyValue train_value_iteration(int num_iters = 30);

private:
    MDP_Ptr mdp_;
    int nx_, nu_;
    float gamma_ = 0.9;
    State x_goal_;
};
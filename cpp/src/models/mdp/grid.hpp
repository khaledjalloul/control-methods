#pragma once

#include "mdp.hpp"

using GridState = std::tuple<int, int>;
enum GridAction
{
    stay,
    top,
    left,
    right,
    bottom
};

class Grid : public MarkovDecisionProcess<GridState, GridAction>
{
private:
    GridState goal_;

public:
    int size;
    GridState x_initial;

    Grid(int size, GridState init, GridState goal);

    int state_to_index(GridState x);

    GridState index_to_state(int index);

    std::tuple<int, Eigen::VectorXd> get_reward_and_trans_prob(int index, GridAction u);

    bool is_done(GridState x);

    GridState play(GridState x_old, GridAction u);
};
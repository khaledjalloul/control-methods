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
public:
    Grid(int size);

    int state_to_index(GridState x);

    GridState index_to_state(int x_index);

    std::tuple<int, Vector> get_reward_and_trans_prob(int x_index, GridAction u, GridState x_goal);

    bool is_done(GridState x, GridState x_goal);

    GridState play(GridState x_old, GridAction u);

private:
    int size_;
};
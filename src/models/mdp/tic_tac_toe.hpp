#pragma once

#include "mdp.hpp"

using TTTState = Eigen::Matrix3i;
using TTTAction = int;

enum TTTStateOptions
{
    _,
    O,
    X
};

class TicTacToe : public MarkovDecisionProcess<TTTState, TTTAction>
{
private:
    int side_ = TTTStateOptions::O, opp_side_ = TTTStateOptions::X;

public:
    TicTacToe();

    int state_to_index(TTTState x);

    TTTState index_to_state(int x_index);

    std::tuple<int, Eigen::VectorXd> get_reward_and_trans_prob(int x_index, TTTAction u, TTTState x_goal = Eigen::Matrix3i::Zero());

    bool is_done(TTTState x, TTTState x_goal = Eigen::Matrix3i::Zero());

    TTTState play(TTTState x_old, TTTAction u);

    std::string display_game(TTTState x);

    TTTState user_play(TTTState);

    void play_match(int num_eps, Eigen::MatrixXd policy);
};
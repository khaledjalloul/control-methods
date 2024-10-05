#include "demos.hpp"

void tic_tac_toe_demo()
{
    auto ttt = std::make_shared<TicTacToe>();
    QLearning<TTTState, TTTAction> mdp(ttt, (int)pow(3, 9), 9, Eigen::Matrix3i::Zero(), Eigen::Matrix3i::Zero());

    mdp.train(70000);
    Matrix policy = mdp.get_policy();
    ttt->play_match(10, policy);
}
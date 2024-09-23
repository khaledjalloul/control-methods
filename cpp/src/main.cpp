#include "controllers/q_learning.hpp"
#include "models/mdp/grid.hpp"

int main()
{
    auto grid = std::make_shared<Grid>(5, GridState{0, 0}, GridState{2, 3});
    QLearning<GridState, GridAction> q_learning(grid, grid->size * grid->size, GridAction::bottom + 1);

    Eigen::MatrixXd Q = q_learning.RL_Q_learning(1000);
    Eigen::MatrixXd V = Q.rowwise().maxCoeff();
    V.resize(grid->size, grid->size);
    std::cout << V << std::endl;

    // TicTacToe ttt;
    // QLearning<TTTState, TTTAction> mdp(&ttt, (int)pow(3, 9), 9);

    // Eigen::MatrixXd Q = mdp.RL_QLearning(70000);
    // Eigen::MatrixXd policy = mdp.getPolicyFromQ(Q);
    // ttt.playMatch(10, policy);

    // return 0;
}
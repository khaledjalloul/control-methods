#include "controllers/q_learning.hpp"
#include "models/mdp/grid.hpp"
#include "models/mdp/tic_tac_toe.hpp"

int main()
{
    int size = 5;
    auto init = GridState{0, 0};
    auto goal = GridState{2, 3};

    auto grid = std::make_shared<Grid>(size);
    QLearning<GridState, GridAction> q_learning(grid, size * size, GridAction::bottom + 1, init, goal);

    q_learning.train(1000);
    Eigen::MatrixXd V = q_learning.get_Q().rowwise().maxCoeff();
    V.resize(size, size);
    std::cout << V << std::endl;

    auto ttt = std::make_shared<TicTacToe>();
    QLearning<TTTState, TTTAction> mdp(ttt, (int)pow(3, 9), 9, Eigen::Matrix3i::Zero(), Eigen::Matrix3i::Zero());

    mdp.train(70000);
    Eigen::MatrixXd policy = mdp.get_policy();
    ttt->play_match(10, policy);

    return 0;
}
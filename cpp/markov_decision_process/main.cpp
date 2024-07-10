#include "mdp.h"

int main()
{
    Grid grid(5, GridState{0, 0}, GridState{2, 3});
    MDP<GridState, GridAction> mdp(&grid, grid.size * grid.size, GridAction::bottom + 1);

    MatrixXd Q = mdp.RL_QLearning(1000);
    MatrixXd V = Q.rowwise().maxCoeff();
    V.resize(grid.size, grid.size);
    cout << V << endl;

    // TicTacToe ttt;
    // MDP<TTTState, TTTAction> mdp(&ttt, (int)pow(3, 9), 9);

    // MatrixXd Q = mdp.RL_QLearning(70000);
    // MatrixXd policy = mdp.getPolicyFromQ(Q);
    // ttt.playMatch(10, policy);

    // return 0;
}
#include "demos.hpp"

void grid_demo()
{
    int size = 5;
    auto init = GridState{0, 0};
    auto goal = GridState{2, 3};

    auto grid = std::make_shared<Grid>(size);
    QLearning<GridState, GridAction> q_learning(grid, size * size, GridAction::bottom + 1, init, goal);

    q_learning.train(1000);
    Vector V = q_learning.get_Q().rowwise().maxCoeff();
    std::cout << V.reshaped(size, size) << std::endl;
}
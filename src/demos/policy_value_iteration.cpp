#include "demos.hpp"

void policy_value_iteration_demo()
{
    int size = 5;
    auto init = GridState{0, 0};
    auto goal = GridState{3, 4};

    auto grid = std::make_shared<Grid>(size);
    PolicyValueIteration<GridState, GridAction> policy_value_iteration(grid, size * size, GridAction::bottom + 1, goal);

    auto policy_V = policy_value_iteration.train_policy_iteration();
    std::cout << "Policy Iteration:\n"
              << policy_V.V.reshaped<Eigen::RowMajor>(size, size) << "\n "
              << std::endl;

    policy_V = policy_value_iteration.train_value_iteration();
    std::cout << "Value Iteration:\n"
              << policy_V.V.reshaped<Eigen::RowMajor>(size, size) << std::endl;
}
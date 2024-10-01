#include <string>

#include "controllers/q_learning.hpp"
#include "controllers/policy_value_iteration.hpp"

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

void tic_tac_toe_demo()
{
    auto ttt = std::make_shared<TicTacToe>();
    QLearning<TTTState, TTTAction> mdp(ttt, (int)pow(3, 9), 9, Eigen::Matrix3i::Zero(), Eigen::Matrix3i::Zero());

    mdp.train(70000);
    Matrix policy = mdp.get_policy();
    ttt->play_match(10, policy);
}

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

int main(int argc, char **argv)
{
    if (argc == 1)
    {
        std::cout << "Please provide an argument:\n- grid (1)\n- tic_tac_toe (2)\n- policy_value_iteration (3)" << std::endl;
    }
    else
    {
        std::string arg = argv[1];
        if (arg == "grid" || arg == "1")
            grid_demo();
        else if (arg == "tic_tac_toe" || arg == "2")
            tic_tac_toe_demo();
        else if (arg == "policy_value_iteration" || arg == "3")
            policy_value_iteration_demo();
    }

    return 0;
}
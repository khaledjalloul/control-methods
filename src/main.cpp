#include <iostream>
#include "demos/demos.hpp"

int main(int argc, char **argv)
{
    if (argc == 1)
    {
        std::cout << "Please provide an argument:\n- grid (1)\n- tic_tac_toe (2)\n- policy_value_iteration (3)\n- mpc (4)" << std::endl;
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
        else if (arg == "mpc" || arg == "4")
            mpc_demo();
    }

    return 0;
}
#include "state_space.hpp"

template <typename State, typename Action>
LTI_StateSpaceModel<State, Action>::LTI_StateSpaceModel(Matrix A, Matrix B) : A_(A), B_(B)
{
    nx_ = A.rows();
    nu_ = B.cols();

    // int num_inputs = sizeof(possible_inputs) / sizeof(*possible_inputs);

    // for (int i = 0; i < num_inputs; i++)
    // {
    //     possible_inputs[i] = -2 + i * 4.0 / num_inputs;
    // }
}

template <typename State, typename Action>
Matrix LTI_StateSpaceModel<State, Action>::x_next(State x, Action u)
{
    return A_ * x + B_ * u;
}

template <typename State, typename Action>
void LTI_StateSpaceModel<State, Action>::find_steady_state(State desired_x_ss, Action desired_u_ss)
{
}

// template <typename State, typename Action>
// float LTI_StateSpaceModel<State, Action>::get_reward(State x, Action u, State x_ss, Action u_ss)
// {
//     return (x - xss).dot(x - xss) + pow(u - uss, 2) * 0.1;
// }

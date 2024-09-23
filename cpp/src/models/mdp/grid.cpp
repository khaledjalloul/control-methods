#include "grid.hpp"

Grid::Grid(int size) : size_(size) {}

int Grid::state_to_index(GridState x)
{
    return std::get<0>(x) * size_ + std::get<1>(x);
}

GridState Grid::index_to_state(int x_index)
{
    return GridState{x_index / size_, x_index % size_};
}

std::tuple<int, Eigen::VectorXd> Grid::get_reward_and_trans_prob(int x_index, GridAction u, GridState x_goal)
{
    GridState x_old = index_to_state(x_index);
    GridState x_next;

    Eigen::VectorXd trans_prob(size_ * size_);

    if (u == GridAction::stay)
        x_next = x_old;
    else if (u == GridAction::top)
    {
        if (std::get<0>(x_old) == 0)
        {
            trans_prob(x_index) = 1;
            return {-1, trans_prob};
        }
        x_next = GridState{std::get<0>(x_old) - 1, std::get<1>(x_old)};
    }
    else if (u == GridAction::left)
    {
        if (std::get<1>(x_old) == 0)
        {
            trans_prob(x_index) = 1;
            return {-1, trans_prob};
        }
        x_next = GridState{std::get<0>(x_old), std::get<1>(x_old) - 1};
    }
    else if (u == GridAction::right)
    {
        if (std::get<1>(x_old) == size_ - 1)
        {
            trans_prob(x_index) = 1;
            return {-1, trans_prob};
        }
        x_next = GridState{std::get<0>(x_old), std::get<1>(x_old) + 1};
    }
    else if (u == GridAction::bottom)
    {
        if (std::get<0>(x_old) == size_ - 1)
        {
            trans_prob(x_index) = 1;
            return {-1, trans_prob};
        }
        x_next = GridState{std::get<0>(x_old) + 1, std::get<1>(x_old)};
    }

    if (x_old == x_goal)
    {
        trans_prob(x_index) = 1;
        return {1, trans_prob};
    }

    int next_index = state_to_index(x_next);
    trans_prob(next_index) = 1;
    return {0, trans_prob};
}

bool Grid::is_done(GridState x, GridState x_goal)
{
    return x == x_goal;
}

GridState Grid::play(GridState x_old, GridAction u)
{
    GridState x_next = x_old;

    if (u == GridAction::top && std::get<0>(x_old) > 0)
        x_next = GridState{std::get<0>(x_old) - 1, std::get<1>(x_old)};
    else if (u == GridAction::left && std::get<1>(x_old) > 0)
        x_next = GridState{std::get<0>(x_old), std::get<1>(x_old) - 1};
    else if (u == GridAction::right && std::get<1>(x_old) < size_ - 1)
        x_next = GridState{std::get<0>(x_old), std::get<1>(x_old) + 1};
    else if (u == GridAction::bottom && std::get<0>(x_old) < size_ - 1)
        x_next = GridState{std::get<0>(x_old) + 1, std::get<1>(x_old)};

    return x_next;
}
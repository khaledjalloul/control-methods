#include "grid.hpp"

Grid::Grid(int size) : size_(size) {}

int Grid::state_to_index(GridState x)
{
    return x.y * size_ + x.x;
}

GridState Grid::index_to_state(int x_index)
{
    return GridState{x_index / size_, x_index % size_};
}

RewardTransProb Grid::get_reward_and_trans_prob(int x_index, GridAction u, GridState x_goal)
{
    GridState x_old = index_to_state(x_index);
    GridState x_next;

    Vector trans_prob = Vector::Zero(size_ * size_);

    if (u == GridAction::stay)
        x_next = x_old;
    else if (u == GridAction::top)
    {
        if (x_old.y == 0)
        {
            trans_prob(x_index) = 1;
            return {-1, trans_prob};
        }
        x_next = GridState{x_old.y - 1, x_old.x};
    }
    else if (u == GridAction::left)
    {
        if (x_old.x == 0)
        {
            trans_prob(x_index) = 1;
            return {-1, trans_prob};
        }
        x_next = GridState{x_old.y, x_old.x - 1};
    }
    else if (u == GridAction::right)
    {
        if (x_old.x == size_ - 1)
        {
            trans_prob(x_index) = 1;
            return {-1, trans_prob};
        }
        x_next = GridState{x_old.y, x_old.x + 1};
    }
    else if (u == GridAction::bottom)
    {
        if (x_old.y == size_ - 1)
        {
            trans_prob(x_index) = 1;
            return {-1, trans_prob};
        }
        x_next = GridState{x_old.y + 1, x_old.x};
    }

    if (x_old.x == x_goal.x && x_old.y == x_goal.y)
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
    return x.x == x_goal.x && x.y == x_goal.y;
}

GridState Grid::play(GridState x_old, GridAction u)
{
    GridState x_next = x_old;

    if (u == GridAction::top && x_old.y > 0)
        x_next = GridState{x_old.y - 1, x_old.x};
    else if (u == GridAction::left && x_old.x > 0)
        x_next = GridState{x_old.y, x_old.x - 1};
    else if (u == GridAction::right && x_old.x < size_ - 1)
        x_next = GridState{x_old.y, x_old.x + 1};
    else if (u == GridAction::bottom && x_old.y < size_ - 1)
        x_next = GridState{x_old.y + 1, x_old.x};

    return x_next;
}
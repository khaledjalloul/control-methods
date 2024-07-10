#include "grid.h"

using namespace std;
using namespace Eigen;

Grid::Grid(int size, GridState init, GridState goal) : size(size), goal(goal)
{
    initialState = init;
}

int Grid::stateToIndex(GridState state)
{
    return get<0>(state) * size + get<1>(state);
}

GridState Grid::indexToState(int index)
{
    return GridState{index / size, index % size};
}

tuple<int, VectorXd> Grid::getRewardAndPTrans(int stateIndex, GridAction u)
{
    GridState oldState = indexToState(stateIndex);
    GridState nextState;

    VectorXd PTrans(size * size);

    if (u == GridAction::stay)
        nextState = oldState;
    else if (u == GridAction::top)
    {
        if (get<0>(oldState) == 0)
        {
            PTrans(stateIndex) = 1;
            return {-1, PTrans};
        }
        nextState = GridState{get<0>(oldState) - 1, get<1>(oldState)};
    }
    else if (u == GridAction::left)
    {
        if (get<1>(oldState) == 0)
        {
            PTrans(stateIndex) = 1;
            return {-1, PTrans};
        }
        nextState = GridState{get<0>(oldState), get<1>(oldState) - 1};
    }
    else if (u == GridAction::right)
    {
        if (get<1>(oldState) == size - 1)
        {
            PTrans(stateIndex) = 1;
            return {-1, PTrans};
        }
        nextState = GridState{get<0>(oldState), get<1>(oldState) + 1};
    }
    else if (u == GridAction::bottom)
    {
        if (get<0>(oldState) == size - 1)
        {
            PTrans(stateIndex) = 1;
            return {-1, PTrans};
        }
        nextState = GridState{get<0>(oldState) + 1, get<1>(oldState)};
    }

    if (oldState == goal)
    {
        PTrans(stateIndex) = 1;
        return {1, PTrans};
    }

    int nextStateIndex = stateToIndex(nextState);
    PTrans(nextStateIndex) = 1;
    return {0, PTrans};
}

bool Grid::isDone(GridState state)
{
    return state == goal;
}

GridState Grid::play(GridState oldState, GridAction u)
{
    GridState nextState = oldState;

    if (u == GridAction::top && get<0>(oldState) > 0)
        nextState = GridState{get<0>(oldState) - 1, get<1>(oldState)};
    else if (u == GridAction::left && get<1>(oldState) > 0)
        nextState = GridState{get<0>(oldState), get<1>(oldState) - 1};
    else if (u == GridAction::right && get<1>(oldState) < size - 1)
        nextState = GridState{get<0>(oldState), get<1>(oldState) + 1};
    else if (u == GridAction::bottom && get<0>(oldState) < size - 1)
        nextState = GridState{get<0>(oldState) + 1, get<1>(oldState)};

    return nextState;
}
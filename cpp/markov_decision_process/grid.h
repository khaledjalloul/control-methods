#include "sim.h"

#ifndef GRID
#define GRID

typedef std::tuple<int, int> GridState;
enum GridAction
{
    stay,
    top,
    left,
    right,
    bottom
};

class Grid : public Sim<GridState, GridAction>
{
private:
    GridState goal;

public:
    int size;

    Grid(int size, GridState init, GridState goal);

    int stateToIndex(GridState state);

    GridState indexToState(int index);

    std::tuple<int, Eigen::VectorXd> getRewardAndPTrans(int stateIndex, GridAction u);

    bool isDone(GridState state);

    GridState play(GridState oldState, GridAction u);
};

#endif
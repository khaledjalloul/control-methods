#include <iostream>
#include <tuple>
#include <Eigen/Dense>

#ifndef SIM_DECLARATION
#define SIM_DECLARATION

template <typename State, typename Action>
class Sim
{
public:
    State initialState;

    virtual int stateToIndex(State state) = 0;

    virtual State indexToState(int index) = 0;

    virtual std::tuple<int, Eigen::VectorXd> getRewardAndPTrans(int stateIndex, Action u) = 0;

    virtual bool isDone(State state) = 0;

    virtual State play(State oldState, Action u) = 0;

    virtual std::tuple<Action, State> randomPlay(State state) = 0;
};

#endif
#include "sim.h"

#ifndef TIC_TAC_TOE
#define TIC_TAC_TOE

typedef Eigen::Matrix3i TTTState;
typedef int TTTAction;

enum TTTStateOptions
{
    _,
    O,
    X
};

class TicTacToe : public Sim<TTTState, TTTAction>
{
private:
    int side = TTTStateOptions::O, oppSide = TTTStateOptions::X;

public:
    TicTacToe();

    int stateToIndex(TTTState state);

    TTTState indexToState(int index);

    std::tuple<int, Eigen::VectorXd> getRewardAndPTrans(int stateIndex, TTTAction u);

    bool isDone(TTTState state);

    TTTState play(TTTState oldState, TTTAction u);

    std::string displayGame(TTTState state);

    TTTState userPlay(TTTState);

    void playMatch(int numEps, Eigen::MatrixXd policy);
};

#endif
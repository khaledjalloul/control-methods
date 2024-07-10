#include "grid.h"
#include "ticTacToe.h"

#ifndef MDP_DEF
#define MDP_DEF

template <typename State, typename Action>
class MDP
{
private:
    Sim<State, Action> *sim;
    int nx, nu, maxStepsPerEpisode = 100;
    float gamma = 0.9, epsilon = 0.9, RL_LearningRate = 0.5;

public:
    MDP(Sim<State, Action> *sim, int nx, int nu);

    Action sampleActionFromQ(int stateIndex, MatrixXd Q, int iter, int numIters);

    MatrixXd getPolicyFromQ(MatrixXd Q);

    MatrixXd RL_QLearning(int numEps);
};

#endif
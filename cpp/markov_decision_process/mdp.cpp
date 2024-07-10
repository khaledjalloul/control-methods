#include "grid.h"
#include "ticTacToe.h"

using namespace std;
using namespace Eigen;

template <typename State, typename Action>
class MDP
{
private:
    Sim<State, Action> *sim;
    int nx, nu, maxStepsPerEpisode = 100;
    float gamma = 0.9, epsilon = 0.7, RL_LearningRate = 0.5;

public:
    MDP(Sim<State, Action> *sim, int nx, int nu) : sim(sim), nx(nx), nu(nu) {}

    Action sampleActionFromQ(int stateIndex, MatrixXd Q, int iter, int numIters)
    {
        if (Q.row(stateIndex).maxCoeff() == 0)
            return Action(rand() % nu);

        if ((rand() / double(RAND_MAX)) <= epsilon - ((double)iter / numIters))
            return Action(rand() % nu);

        int argmax;
        Q.row(stateIndex).maxCoeff(&argmax);
        return Action(argmax);
    }

    MatrixXd getPolicyFromQ(MatrixXd Q)
    {
        MatrixXd policy = MatrixXd::Zero(nx, nu);

        for (int x = 0; x < nx; x++)
        {
            if ((Q.row(x).array() != 0).any())
            {
                int u;
                Q.row(x).maxCoeff(&u);
                policy(x, u) = 1;
            }
        }

        return policy;
    }

    MatrixXd RL_QLearning(int numEps)
    {
        MatrixXd Q = MatrixXd::Zero(nx, nu);

        for (int i = 0; i < numEps; i++)
        {
            cout << "Training: " << ((float)i / numEps) * 100 << "%" << endl;

            State state = sim->initialState;
            MatrixXd QOld = Q;

            for (int j = 0; j < maxStepsPerEpisode; j++)
            {
                int stateIndex = sim->stateToIndex(state);
                Action u = sampleActionFromQ(stateIndex, Q, i, numEps);

                tuple<int, MatrixXd> rewardAndPTrans = sim->getRewardAndPTrans(stateIndex, u);
                int reward = get<0>(rewardAndPTrans);

                State nextState = sim->play(state, u);

                int nextStateIndex = sim->stateToIndex(nextState);

                Q(stateIndex, u) = (1 - RL_LearningRate) * Q(stateIndex, u) + RL_LearningRate * (reward + gamma * Q.row(nextStateIndex).maxCoeff());

                if (sim->isDone(state))
                    break;

                state = nextState;
            }
        }

        return Q;
    }
};

int main()
{
    // Grid grid(5, GridState{0, 0}, GridState{3, 3});
    // MDP<GridState, GridAction> mdp(&grid, grid.size * grid.size, GridAction::bottom + 1);

    // MatrixXd Q = mdp.RL_QLearning(1000);
    // MatrixXd V = Q.rowwise().maxCoeff();
    // V.resize(grid.size, grid.size);
    // cout << V << endl;

    TicTacToe ttt;
    MDP<TTTState, TTTAction> mdp(&ttt, (int)pow(3, 9), 9);

    MatrixXd Q = mdp.RL_QLearning(70000);
    MatrixXd policy = mdp.getPolicyFromQ(Q);
    ttt.playMatch(10, policy);

    return 0;
}
#include "grid.h"

using namespace std;
using namespace Eigen;

template <typename State, typename Action>
class MDP
{
private:
    Sim<State, Action> *sim;
    int nx, nu, maxStepsPerEpisode = 100;
    float gamma = 0.9, epsilon = 0.7, RL_LearningRate = 0.5;
    VectorXd V;
    MatrixXd policy;

public:
    MDP(Sim<State, Action> *sim, int nx, int nu) : sim(sim), nx(nx), nu(nu)
    {
        V = VectorXd::Random(nx);
        policy = MatrixXd::Ones(nx, nu) / nu;
    }

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

    MatrixXd RL_QLearning(int numEps)
    {
        MatrixXd Q = MatrixXd::Zero(nx, nu);

        for (int i = 0; i < numEps; i++)
        {
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
    int size = 5;
    Grid grid(size, GridState{0, 0}, GridState{3, 3});
    MDP<GridState, GridAction> mdp(&grid, size * size, GridAction::bottom + 1);

    MatrixXd Q = mdp.RL_QLearning(1000);
    MatrixXd V = Q.rowwise().maxCoeff();
    V.resize(grid.size, grid.size);
    cout << V << endl;

    return 0;
}
#include "mdp.h"

template <typename State, typename Action>
MDP<State, Action>::MDP(Sim<State, Action> *sim, int nx, int nu) : sim(sim), nx(nx), nu(nu) {}

template <typename State, typename Action>
Action MDP<State, Action>::sampleActionFromQ(int stateIndex, MatrixXd Q, int iter, int numIters)
{
    if (Q.row(stateIndex).maxCoeff() == 0)
        return Action(rand() % nu);

    if ((rand() / double(RAND_MAX)) <= epsilon - ((double)iter / numIters))
        return Action(rand() % nu);

    int argmax;
    Q.row(stateIndex).maxCoeff(&argmax);
    return Action(argmax);
}

template <typename State, typename Action>
MatrixXd MDP<State, Action>::getPolicyFromQ(MatrixXd Q)
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

template <typename State, typename Action>
MatrixXd MDP<State, Action>::RL_QLearning(int numEps)
{
    MatrixXd Q = MatrixXd::Zero(nx, nu);

    for (int i = 0; i < numEps; i++)
    {
        cout << "Training: " << ((float)i / numEps) * 100 << "%" << endl;

        State state = sim->initialState;

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

template class MDP<GridState, GridAction>;
template class MDP<TTTState, TTTAction>;
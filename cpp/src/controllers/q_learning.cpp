#include "q_learning.hpp"

template <typename State, typename Action>
QLearning<State, Action>::QLearning(std::shared_ptr<MarkovDecisionProcess<State, Action>> mdp, int nx, int nu) : mdp_(std::move(mdp)), nx_(nx), nu_(nu) {}

template <typename State, typename Action>
Action QLearning<State, Action>::sample_action_from_Q(int index, Eigen::MatrixXd Q, int iter, int num_iters)
{
    if (Q.row(index).maxCoeff() == 0)
        return Action(rand() % nu_);

    if ((rand() / double(RAND_MAX)) <= epsilon_ - ((double)iter / num_iters))
        return Action(rand() % nu_);

    int argmax;
    Q.row(index).maxCoeff(&argmax);
    return Action(argmax);
}

template <typename State, typename Action>
Eigen::MatrixXd QLearning<State, Action>::get_policy_from_Q(Eigen::MatrixXd Q)
{
    Eigen::MatrixXd policy = Eigen::MatrixXd::Zero(nx_, nu_);

    for (int x = 0; x < nx_; x++)
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
Eigen::MatrixXd QLearning<State, Action>::RL_Q_learning(int numEps)
{
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(nx_, nu_);

    for (int i = 0; i < numEps; i++)
    {
        std::cout << "Training: " << ((float)i / numEps) * 100 << "%" << std::endl;

        State x = mdp_->x_initial;

        for (int j = 0; j < max_steps_per_episode_; j++)
        {
            int index = mdp_->state_to_index(x);
            Action u = sample_action_from_Q(index, Q, i, numEps);

            std::tuple<int, Eigen::MatrixXd> reward_and_trans_prob = mdp_->get_reward_and_trans_prob(index, u);
            int reward = std::get<0>(reward_and_trans_prob);

            State x_next = mdp_->play(x, u);

            int next_index = mdp_->state_to_index(x_next);

            Q(index, u) = (1 - RL_learning_rate_) * Q(index, u) + RL_learning_rate_ * (reward + gamma_ * Q.row(next_index).maxCoeff());

            if (mdp_->is_done(x))
                break;

            x = x_next;
        }
    }

    return Q;
}

#include "../models/mdp/grid.hpp"
template class QLearning<GridState, GridAction>;

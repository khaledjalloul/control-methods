#include "q_learning.hpp"

template <typename State, typename Action>
QLearning<State, Action>::QLearning(std::shared_ptr<MarkovDecisionProcess<State, Action>> mdp, int nx, int nu, State x_initial, State x_goal) :
    mdp_(std::move(mdp)), nx_(nx), nu_(nu), x_initial_(x_initial), x_goal_(x_goal)
{
    Q_ = Eigen::MatrixXd::Zero(nx_, nu_);
}

template <typename State, typename Action>
Eigen::MatrixXd QLearning<State, Action>::get_Q()
{
    return Q_;
}

template <typename State, typename Action>
Action QLearning<State, Action>::sample_action(int index, int iter, int num_iters)
{
    if (Q_.row(index).maxCoeff() == 0)
        return Action(rand() % nu_);

    if ((rand() / double(RAND_MAX)) <= epsilon_ - ((double)iter / num_iters))
        return Action(rand() % nu_);

    int argmax;
    Q_.row(index).maxCoeff(&argmax);
    return Action(argmax);
}

template <typename State, typename Action>
Eigen::MatrixXd QLearning<State, Action>::get_policy()
{
    Eigen::MatrixXd policy = Eigen::MatrixXd::Zero(nx_, nu_);

    for (int x = 0; x < nx_; x++)
    {
        if ((Q_.row(x).array() != 0).any())
        {
            int u;
            Q_.row(x).maxCoeff(&u);
            policy(x, u) = 1;
        }
    }

    return policy;
}

template <typename State, typename Action>
void QLearning<State, Action>::train(int num_eps)
{

    for (int i = 0; i < num_eps; i++)
    {
        std::cout << "Training: " << ((float)i / num_eps) * 100 << "%" << std::endl;

        State x = x_initial_;

        for (int j = 0; j < max_steps_per_episode_; j++)
        {
            int index = mdp_->state_to_index(x);
            Action u = sample_action(index, i, num_eps);

            auto reward_and_trans_prob = mdp_->get_reward_and_trans_prob(index, u, x_goal_);
            int reward = std::get<0>(reward_and_trans_prob);

            State x_next = mdp_->play(x, u);

            int next_index = mdp_->state_to_index(x_next);

            Q_(index, u) = (1 - RL_learning_rate_) * Q_(index, u) + RL_learning_rate_ * (reward + gamma_ * Q_.row(next_index).maxCoeff());

            if (mdp_->is_done(x, x_goal_))
                break;

            x = x_next;
        }
    }
}

#include "../models/mdp/grid.hpp"
#include "../models/mdp/tic_tac_toe.hpp"
template class QLearning<GridState, GridAction>;
template class QLearning<TTTState, TTTAction>;

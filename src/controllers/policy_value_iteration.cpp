#include "policy_value_iteration.hpp"

template <class State, class Action>
PolicyValueIteration<State, Action>::PolicyValueIteration(MDP_Ptr mdp, int nx, int nu, State x_goal)
    : mdp_(std::move(mdp)), nx_(nx), nu_(nu), x_goal_(x_goal)
{
}

template <class State, class Action>
Vector PolicyValueIteration<State, Action>::evaluate_policy(Matrix policy)
{
    Vector R = Vector::Zero(nx_);
    Matrix P = Matrix::Zero(nx_, nx_);

    for (int u = 0; u < nu_; u++)
    {
        for (int x = 0; x < nx_; x++)
        {
            auto reward_and_trans_prob = mdp_->get_reward_and_trans_prob(x, Action(u), x_goal_);

            R(x) = R(x) + policy(x, u) * std::get<0>(reward_and_trans_prob);
            P.row(x) += policy(x, u) * std::get<1>(reward_and_trans_prob);
        }
    }

    Vector V = (Matrix::Identity(nx_, nx_) - gamma_ * P).inverse() * R;
    return V;
}

template <class State, class Action>
std::tuple<Matrix, Vector> PolicyValueIteration<State, Action>::improve_policy(Vector V)
{
    Matrix policy = Matrix::Ones(nx_, nu_) / nu_;
    Vector V_new = Vector::Zero(nx_);

    for (int x = 0; x < nx_; x++)
    {
        Vector scores = Vector::Zero(nu_);
        for (int u = 0; u < nu_; u++)
        {
            auto reward_and_trans_prob = mdp_->get_reward_and_trans_prob(x, Action(u), x_goal_);

            scores(u) = std::get<0>(reward_and_trans_prob) + gamma_ * std::get<1>(reward_and_trans_prob).dot(V);

            policy.row(x).setZero();
            int argmax;
            V_new(x) = scores.maxCoeff(&argmax);
            policy(x, argmax) = 1;
        }
    }

    return std::make_tuple(policy, V_new);
}

template <class State, class Action>
std::tuple<Matrix, Vector> PolicyValueIteration<State, Action>::train_policy_iteration(int num_iters)
{
    Matrix policy = Matrix::Ones(nx_, nu_) / nu_;
    Vector V = Vector::Random(nx_);
    Vector V_new = Vector::Zero(nx_);

    for (int i = 0; i < num_iters; i++)
    {
        V_new = evaluate_policy(policy);

        auto diff = ((Vector)(V_new - V)).lpNorm<Eigen::Infinity>();
        V = V_new;

        auto policy_and_V_new = improve_policy(V);
        policy = std::get<0>(policy_and_V_new);

        if (diff < 0.1)
            break;
    }

    return std::make_tuple(policy, V);
}

template <class State, class Action>
std::tuple<Matrix, Vector> PolicyValueIteration<State, Action>::train_value_iteration(int num_iters)
{
    Matrix policy = Matrix::Ones(nx_, nu_) / nu_;
    Vector V = Vector::Zero(nx_);

    for (int i = 0; i < num_iters; i++)
    {
        auto policy_and_V_new = improve_policy(V);

        auto diff = ((Vector)(std::get<1>(policy_and_V_new) - V)).lpNorm<Eigen::Infinity>();
        policy = std::get<0>(policy_and_V_new);
        V = std::get<1>(policy_and_V_new);

        if (diff < 0.1)
            break;
    }

    return std::make_tuple(policy, V);
}

template class PolicyValueIteration<GridState, GridAction>;
template class PolicyValueIteration<TTTState, TTTAction>;
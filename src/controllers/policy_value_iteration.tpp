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

            R(x) = R(x) + policy(x, u) * reward_and_trans_prob.reward;
            P.row(x) += policy(x, u) * reward_and_trans_prob.trans_prob;
        }
    }

    Vector V = (Matrix::Identity(nx_, nx_) - gamma_ * P).inverse() * R;
    return V;
}

template <class State, class Action>
PolicyValue PolicyValueIteration<State, Action>::improve_policy(Vector V)
{
    Matrix policy = Matrix::Ones(nx_, nu_) / nu_;
    Vector V_new = Vector::Zero(nx_);

    for (int x = 0; x < nx_; x++)
    {
        Vector scores = Vector::Zero(nu_);
        for (int u = 0; u < nu_; u++)
        {
            auto reward_and_trans_prob = mdp_->get_reward_and_trans_prob(x, Action(u), x_goal_);

            scores(u) = reward_and_trans_prob.reward + gamma_ * reward_and_trans_prob.trans_prob.dot(V);

            policy.row(x).setZero();
            int argmax;
            V_new(x) = scores.maxCoeff(&argmax);
            policy(x, argmax) = 1;
        }
    }

    return {policy, V_new};
}

template <class State, class Action>
PolicyValue PolicyValueIteration<State, Action>::train_policy_iteration(int num_iters)
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
        policy = policy_and_V_new.policy;

        if (diff < 0.1)
            break;
    }

    return {policy, V};
}

template <class State, class Action>
PolicyValue PolicyValueIteration<State, Action>::train_value_iteration(int num_iters)
{
    Matrix policy = Matrix::Ones(nx_, nu_) / nu_;
    Vector V = Vector::Zero(nx_);

    for (int i = 0; i < num_iters; i++)
    {
        auto policy_and_V_new = improve_policy(V);

        auto diff = ((Vector)(policy_and_V_new.V - V)).lpNorm<Eigen::Infinity>();
        policy = policy_and_V_new.policy;
        V = policy_and_V_new.V;

        if (diff < 0.1)
            break;
    }

    return {policy, V};
}
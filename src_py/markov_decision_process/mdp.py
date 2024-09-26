import numpy as np
from tictactoe import TicTacToe
from grid import Grid
import os


class MDP:
    def __init__(self):
        self.sim = TicTacToe()
        self.nx = 3 ** 9
        self.nu = 9
        self.initial_state = lambda: np.zeros((3, 3), dtype=np.int8)
        self.max_steps_per_episode = 100

        # self.sim = Grid()
        # self.nx = self.sim.size ** 2
        # self.nu = 5
        # self.initial_state = lambda: self.sim.index_to_state(
        #     np.random.choice(self.nx))
        # self.max_steps_per_episode = 10

        self.gamma = 0.9
        self.epsilon = 0.7
        self.RL_learning_rate = 0.5

        self.V = np.random.randn(self.nx)
        self.policy = np.ones((self.nx, self.nu)) * (1 / self.nu)

    def evaluate_policy(self, policy):
        R = np.zeros(self.nx)
        P = np.zeros((self.nx, self.nx))

        for u in range(self.nu):
            for x in range(self.nx):
                reward, P_trans = self.sim.get_reward_P_trans(x, u)

                P[x] += policy[x, u] * P_trans
                R[x] += policy[x, u] * reward

        V = np.linalg.solve(np.eye(self.nx) - self.gamma * P, R)
        return V

    def improve_policy(self, V):
        policy = np.zeros((self.nx, self.nu))
        V_new = np.zeros(self.nx)

        for x in range(self.nx):
            scores = np.zeros(self.nu)
            for u in range(self.nu):
                reward, P_trans = self.sim.get_reward_P_trans(x, u)

                scores[u] = reward + self.gamma * np.dot(P_trans, V)

            policy[x, :] = np.zeros(self.nu)
            policy[x, np.argmax(scores)] = 1
            V_new[x] = np.max(scores)

        return policy, V_new

    def train_policy_iteration(self, iters=10):
        for _ in range(iters):
            V_new = self.evaluate_policy(self.policy)

            diff = np.linalg.norm(V_new - self.V, np.inf)
            self.V = np.copy(V_new)

            self.policy, _ = self.improve_policy(self.V)

            print(diff)
            if diff < 0.1:
                break

    def train_value_iteration(self, iters=30):
        for _ in range(iters):
            self.policy, V_new = self.improve_policy(self.V)

            diff = np.linalg.norm(V_new - self.V, np.inf)
            self.V = np.copy(V_new)

            print(diff)
            if diff < 0.1:
                break

    def sample_action(self, state, iter, num_iters):
        if np.random.rand() <= self.epsilon - (iter / num_iters):
            return np.random.choice(np.arange(self.nu))
        else:
            state_index = self.sim.state_to_index(state)
            return np.argmax(self.policy[state_index])

    def collect_Q_from_episodes(self, num_eps, iter, num_iters):
        Q = np.zeros((self.nx, self.nu))
        count_visited = np.zeros((self.nx, self.nu))

        for _ in range(num_eps):
            state = self.initial_state()
            states = []
            actions = []
            rewards = []

            for _ in range(self.max_steps_per_episode):
                u = self.sample_action(state, iter, num_iters)
                state_index = self.sim.state_to_index(state)
                reward, _ = self.sim.get_reward_P_trans(state_index, u)

                states.append(state_index)
                actions.append(u)
                rewards.append(reward)

                if self.sim.is_done(state):
                    break

                state = self.sim.play(state, u)

            T = len(states)
            for k in range(T):
                gamma_vector = np.power(
                    np.ones(T - k) * self.gamma, np.arange(T - k))
                count_visited[states[k], actions[k]] += 1
                Q[states[k], actions[k]] += np.dot(rewards[k:], gamma_vector)

        Q = np.where(count_visited > 0, Q / count_visited, 0)

        return Q

    def get_policy_from_Q(self, Q):
        policy = np.copy(self.policy)
        for x in range(self.nx):
            if np.any(Q[x] != 0):
                u_i = np.argwhere(Q[x] == np.max(Q[x]))
                policy[x] = np.zeros(self.nu)
                policy[x, u_i] = 1
                policy[x] = policy[x] / np.sum(policy[x])

        return policy

    def V_from_Q(self, Q):
        V = np.zeros(self.nx)

        for x in range(self.nx):
            V[x] = np.dot(self.policy[x], Q[x])

        return V

    def RL_monte_carlo(self, num_iters, num_eps):
        V = np.random.randn(self.nx)
        Q = np.zeros((self.nx, self.nu))

        for i in range(num_iters):
            Q = self.collect_Q_from_episodes(num_eps, i, num_iters)
            self.policy = self.get_policy_from_Q(Q)

            V_new = self.V_from_Q(Q)

            print(Q)
            print(i, np.linalg.norm(V_new - V, np.inf))
            print(V_new)

            V = np.copy(V_new)

        return Q

    def sample_action_from_Q(self, state, Q, iter, num_iters):
        if np.max(Q[state]) == 0:
            return np.random.choice(np.arange(self.nu))
        if np.random.rand() <= self.epsilon - (iter / num_iters):
            return np.random.choice(np.arange(self.nu))
        else:
            return np.argmax(Q[state])

    def RL_SARSA(self, num_eps):
        Q = np.zeros((self.nx, self.nu))

        for i in range(num_eps):
            state = self.initial_state()
            state_index = self.sim.state_to_index(state)
            u = self.sample_action_from_Q(state_index, Q, i, num_eps)

            Q_old = np.copy(Q)

            for _ in range(self.max_steps_per_episode):
                state_index = self.sim.state_to_index(state)
                reward, _ = self.sim.get_reward_P_trans(state_index, u)

                state_next = self.sim.play(state, u)
                state_next_index = self.sim.state_to_index(state_next)
                u_next = self.sample_action_from_Q(
                    state_next_index, Q, i, num_eps)

                Q[state_index, u] = (1 - self.RL_learning_rate) * Q[state_index, u] \
                    + self.RL_learning_rate * \
                    (reward + self.gamma * Q[state_next_index, u_next])

                if self.sim.is_done(state):
                    break

                u = u_next
                state = np.copy(state_next)

            print(i, np.linalg.norm(np.mean(Q_old - Q, axis=1), np.inf))

        return Q

    def RL_Q_learning(self, num_eps):
        Q = np.zeros((self.nx, self.nu))

        for i in range(num_eps):
            state = self.initial_state()
            Q_old = np.copy(Q)

            for _ in range(self.max_steps_per_episode):
                state_index = self.sim.state_to_index(state)
                u = self.sample_action_from_Q(state_index, Q, i, num_eps)

                self.sim.print(state)
                reward, _ = self.sim.get_reward_P_trans(state_index, u)
                state_next = self.sim.play(state, u)
                state_next_index = self.sim.state_to_index(state_next)

                Q[state_index, u] = (1 - self.RL_learning_rate) * Q[state_index, u] \
                    + self.RL_learning_rate * \
                    (reward + self.gamma * np.max(Q[state_next_index]))

                if self.sim.is_done(state):
                    break

                state = np.copy(state_next)

            print(i, np.linalg.norm(np.mean(Q_old - Q, axis=1), np.inf))

        return Q

    def play(self, num_eps):
        for _ in range(num_eps):
            state = np.zeros((3, 3), dtype=np.int8)

            while not self.sim.is_done(state):
                state = self.sim.user_play(state, 2)
                self.sim.print(state, with_numbers=False)

                if self.sim.is_done(state):
                    break

                state_index = self.sim.state_to_index(state)
                u = np.argmax(self.policy[state_index])

                state = self.sim.play(state, u)
                self.sim.print(state, with_numbers=False)
                print("-------")

            print("DONE")


if __name__ == '__main__':
    mdp = MDP()

    # mdp.train_policy_iteration(10)
    # mdp.train_value_iteration(30)
    # mdp.RL_monte_carlo(50, 2000)

    path = "markov_decision_process/out/tic_tac_toe_policy.npy"
    if os.path.isfile(path):
        mdp.policy = np.load(path)
        mdp.play(30)
    else:
        Q = mdp.RL_Q_learning(70000)
        policy = mdp.get_policy_from_Q(Q)
        np.save(path, policy)
        print(mdp.V_from_Q(Q))

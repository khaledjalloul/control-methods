import numpy as np


class Grid:
    def __init__(self):
        self.action_options = ['stay', 'top', 'left', 'right', 'bottom']

        self.size = 6
        self.goal = (4, 3)

    def state_to_index(self, state):
        state_num = state[0] * self.size + state[1]
        return state_num

    def index_to_state(self, n):
        state = (n // self.size, n % self.size)
        return state

    def get_reward_P_trans(self, state_index, u):
        old_state = self.index_to_state(state_index)
        P_trans = np.zeros(self.size * self.size)

        if u == 0:
            next_state = old_state
        elif u == 1:
            if old_state[0] == 0:
                P_trans[state_index] = 1
                return -1, P_trans
            next_state = (old_state[0] - 1, old_state[1])
        elif u == 2:
            if old_state[1] == 0:
                P_trans[state_index] = 1
                return -1, P_trans
            next_state = (old_state[0], old_state[1] - 1)
        elif u == 3:
            if old_state[1] == self.size - 1:
                P_trans[state_index] = 1
                return -1, P_trans
            next_state = (old_state[0], old_state[1] + 1)
        elif u == 4:
            if old_state[0] == self.size - 1:
                P_trans[state_index] = 1
                return -1, P_trans
            next_state = (old_state[0] + 1, old_state[1])

        if old_state[0] == self.goal[0] and old_state[1] == self.goal[1]:
            P_trans[state_index] = 1
            return 1, P_trans

        next_state_index = self.state_to_index(next_state)
        P_trans[next_state_index] = 1

        return 0, P_trans

    def is_done(self, state):
        return state[0] == self.goal[0] and state[1] == self.goal[1]

    def random_play(self, state):
        u = np.random.choice(np.arange(5))
        return u, self.play(state, u)

    def play(self, old_state, u):
        next_state = old_state

        if u == 1 and old_state[0] > 0:
            next_state = (old_state[0] - 1, old_state[1])
        elif u == 2 and old_state[1] > 0:
            next_state = (old_state[0], old_state[1] - 1)
        elif u == 3 and old_state[1] < self.size - 1:
            next_state = (old_state[0], old_state[1] + 1)
        elif u == 4 and old_state[0] < self.size - 1:
            next_state = (old_state[0] + 1, old_state[1])

        return next_state

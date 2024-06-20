import numpy as np


class TicTacToe:
    def __init__(self):
        self.state_options = ['_', 'O', 'X']

    def state_to_index(self, state):
        state_num = "".join([str(x) for x in state.reshape(-1).tolist()])
        state_num = int(state_num, 3)
        return state_num

    def index_to_state(self, n):
        state = []

        for i in range(9):
            cur = n // 3 ** (8 - i)
            state.append(cur)

            n = n - cur * (3 ** (8 - i))

        return np.reshape(state, (3, 3))

    def print(self, state, with_numbers=False):
        out = ""

        for i, row in enumerate(state):
            for j, col in enumerate(row):
                if col == 0 and with_numbers:
                    out += str((i * 3 + j) + 1) + " "
                else:
                    out += self.state_options[col] + " "
            if i < 2:
                out += "\n"

        print(out)

    def is_done(self, state):

        if np.all(state != 0):
            return True

        for i in range(3):
            if np.all(state[i] == state[i, 0]) and state[i, 0] != 0:
                return True
            if np.all(state[:, i] == state[0, i]) and state[0, i] != 0:
                return True

        if np.all(np.diagonal(state) == state[0, 0]) and state[0, 0] != 0:
            return True

        if np.all(np.diagonal(np.fliplr(state)) == state[0, 2]) and state[0, 2] != 0:
            return True

        return False

    def get_reward_P_trans(self, state_index, u, side=1):
        state = self.index_to_state(state_index)
        reward = 0
        loss_factor = 10
        opp_side = 2 if side == 1 else 1
        P_trans = np.zeros(3 ** 9)

        if self.is_done(state):
            P_trans[state_index] = 1

        elif np.count_nonzero(state == opp_side) > np.count_nonzero(state == side):
            i = u // 3
            j = u % 3

            if state[i, j] != 0:
                P_trans[state_index] = 1
                reward = -1
            else:
                new_state = np.copy(state)
                new_state[i, j] = side
                P_trans[self.state_to_index(new_state)] = 1

        else:
            blanks = np.argwhere(state == 0)
            blanks_count = np.shape(blanks)[0]
            blank_prob = (1 / blanks_count) if blanks_count > 0 else 0

            for blank in blanks:
                blank_state = np.copy(state)
                blank_state[blank[0], blank[1]] = opp_side
                blank_state_index = self.state_to_index(blank_state)
                P_trans[blank_state_index] = blank_prob

        for i in range(3):
            row = state[i, :]
            col = state[:, i]

            if np.all(row == opp_side) or np.all(col == opp_side):
                return - loss_factor, P_trans
            if np.all(row == side) or np.all(col == side):
                return 1, P_trans

        diag = np.diagonal(state)
        if np.all(diag == opp_side):
            return - loss_factor, P_trans
        if np.all(diag == side):
            return 1, P_trans

        diag2 = np.diagonal(np.fliplr(state))
        if np.all(diag2 == opp_side):
            return - loss_factor, P_trans
        if np.all(diag2 == side):
            return 1, P_trans

        return reward, P_trans

    def play(self, state, u, side=1):
        opp_side = 2 if side == 1 else 1
        state = np.copy(state)

        if self.is_done(state):
            return state

        if np.count_nonzero(state == opp_side) > np.count_nonzero(state == side):
            i = u // 3
            j = u % 3

            if state[i, j] == 0:
                state[i, j] = side

        else:
            u = np.random.choice(np.arange(9))
            i = u // 3
            j = u % 3

            while state[i, j] != 0:
                u = np.random.choice(np.arange(9))
                i = u // 3
                j = u % 3

            state[i, j] = opp_side

        return state

    def user_play(self, state, side=1):
        try:
            state = np.copy(state)

            u = input("Position (1 to 9): ")
            u = int(u) - 1

            i = u // 3
            j = u % 3

            while u < 0 or u > 8 or state[i, j] != 0:
                u = input("Position (1 to 9): ")
                u = int(u) - 1

                i = u // 3
                j = u % 3

            state[i, j] = side
            return state

        except ValueError:
            return self.user_play(state, side)

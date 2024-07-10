import cvxpy as cp
import numpy as np

class MPC_Controller:
    def __init__(self, model):
        self.K = 10
        
        self.Q = np.eye(model.nx)
        self.R = np.eye(model.nu)
        self.slack_cost = 300
                
    def step(self, model, x0, x_ss, u_ss, delay = 0, prev_inputs = [0], current_index = 0):        
        for k in range(delay):
            u0_index = current_index - delay + k
            u0 = prev_inputs[:, u0_index] if u0_index >= 0 else np.zeros(model.nu)
            x0 = model.x_next(x0, u0)
            
        x = cp.Variable((model.nx, self.K + 1))
        u = cp.Variable((model.nu, self.K))
        s = cp.Variable(self.K)

        cost = 0
        constraints = [x[:, 0] == x0,
                       x[:, -1] == x_ss,
                       u[:, -1] == u_ss]

        for k in range(self.K):
            cost += cp.quad_form((x[:, k] - x_ss), self.Q) \
                    + cp.quad_form((u[:, k] - u_ss), self.R) \
                    + self.slack_cost * cp.abs(s[k])
            
            constraints += [x[:, k+1] == model.A @ x[:, k] + model.B @ u[:, k],
                            cp.norm_inf(u[:, k]) <= 3 + s[k],
                            s[k] >= 0]
    
        problem = cp.Problem(cp.Minimize(cost), constraints)

        result = problem.solve()

        if np.isinf(result):
            return np.zeros(model.nu)

        return u.value[:, 0]
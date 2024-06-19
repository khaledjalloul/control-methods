import numpy as np
import cvxpy as cp

class DeePC_Controller:
    def __init__(self, model, delay):
        self.T_prev = 6
        self.T_fut = 15
        self.L = self.T_prev + self.T_fut
        self.T = 100
        
        self.Q = np.eye(model.ny * self.T_fut)
        self.R = np.eye(model.nu * self.T_fut) * 0.1
        self.slack_cost = 1000
        
        u_sim = 2 * np.random.randn(model.nu, self.T)
        x_sim = np.zeros((model.nx, self.T+1))
        y_sim = np.zeros((model.ny, self.T))
        
        for i in range(self.T):
            u_index = i - delay
            u = u_sim[:, u_index] if u_index >= 0 else np.zeros(model.nu)
            x_sim[:, i+1] = model.x_next(x_sim[:, i], u)
            y_sim[:, i] = model.y(x_sim[:, i])
        
        self.num_hankel_columns = self.T - self.L + 1
        
        self.Hu = np.zeros((model.nu * self.L, self.num_hankel_columns))
        self.Hy = np.zeros((model.ny * self.L, self.num_hankel_columns))
        
        for i in range(self.num_hankel_columns):
            self.Hu[:, i] = u_sim[:, i:i+self.L].T.reshape(-1)
            self.Hy[:, i] = y_sim[:, i:i+self.L].T.reshape(-1)          
    
        
    def step(self, model, x0, x_ss, u_ss, u_past_, y_past_, delay = 0, prev_inputs = [0], current_index = 0):
        U_p = self.Hu[:model.nu*self.T_prev]
        U_f = self.Hu[-model.nu*self.T_fut:]
        Y_p = self.Hy[:model.ny*self.T_prev]
        Y_f = self.Hy[-model.ny*self.T_fut:]
        
        u_past = np.zeros((model.nu, self.T_prev))
        y_past = np.zeros((model.ny, self.T_prev))
        
        for i in range(self.T_prev):
            past_index = current_index - self.T_prev + i
            if past_index >= 0:
                u_past[:, i] = u_past_[:, past_index]
                y_past[:, i] = y_past_[:, past_index]
                
        u_past = u_past.T.reshape(-1)
        y_past = y_past.T.reshape(-1)
                
        y_ref = model.y(x_ss)
        y_ref = np.tile(y_ref, self.T_fut)
        u_ref = np.tile(u_ss, self.T_fut)
        
        g = cp.Variable(self.num_hankel_columns)
        u = cp.Variable(model.nu*self.T_fut)
        y = cp.Variable(model.ny*self.T_fut)
        s = cp.Variable(model.nu*self.T_fut)
        
        cost = cp.quad_form(y - y_ref, self.Q) + cp.quad_form(u - u_ref, self.R) + cp.norm1(s)
        
        constraints = [
            U_p @ g == u_past,
            U_f @ g == u,
            Y_p @ g == y_past,
            Y_f @ g == y,
            cp.norm_inf(u) <= 3,
            s >= 0
        ]
        
        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        result = prob.solve()
        
        if np.isinf(result):
            return np.zeros(model.nu)
        
        return u.value[:model.nu]
    
if __name__ == '__main__':
    from model import LinearStateSpaceModel
    
    model = LinearStateSpaceModel(np.eye(2), np.eye(2))
    DeePC_Controller(model)
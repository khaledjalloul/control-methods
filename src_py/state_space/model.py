import numpy as np
import cvxpy as cp

class LinearStateSpaceModel:
    def __init__(self, A, B, C = None):
        self.A = A
        self.nx = np.shape(A)[0]
        
        self.B = B
        self.nu = np.shape(B)[1]
        
        if C is None:
            self.C = np.eye(self.nx)
            self.ny = self.nx
        else:
            self.C = C
            self.ny = np.shape(C)[0]
        
        
    def x_next(self, x, u):
        return self.A @ x + self.B @ u
    
    def y(self, x):
        return self.C @ x
    
    def find_steady_state(self, desired_x_ss, desired_u_ss = None):
        x_ss = cp.Variable(self.nx)
        u_ss = cp.Variable(self.nu)
        
        cost = cp.quad_form(x_ss - desired_x_ss, np.eye(self.nx))
        
        if desired_u_ss is not None:
            cost += cp.quad_form(u_ss - desired_u_ss, np.eye(self.nu))
        
        constraints = [x_ss == self.A @ x_ss + self.B @ u_ss]
        
        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        result = prob.solve()
        
        if not np.isinf(result):
            return x_ss.value, u_ss.value

if __name__ == '__main__':
    model = LinearStateSpaceModel()
    
    x_ss, u_ss = model.find_steady_state([2, 7], [1, 0])
    
    print(x_ss, u_ss)
    print(model.x_next(x_ss, u_ss))
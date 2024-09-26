import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from model import LinearStateSpaceModel
from mpc import MPC_Controller
from deepc import DeePC_Controller

A = np.array([[1, 0.5], [-0.2, 1]])
B = np.array([[1, 0.3], [-0.1, 1]])
# A = np.array([[1, 0], [0, 1]])
# B = np.array([[1, 0], [0, 1]])

model = LinearStateSpaceModel(A, B)

T = 20
delay = 5
x_ss = [3, 7]
x_ss, u_ss = model.find_steady_state(x_ss)

x_sim = np.zeros((model.nx, T+1))
u_sim = np.zeros((model.nu, T))
x_sim[:, 0] = [-11., 9]

mpc = MPC_Controller(model)

ax1: Axes
ax2: Axes
ax3: Axes
ax4: Axes
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

for i in range(T):
    u_sim[:, i] = mpc.step(model, x_sim[:, i], x_ss, u_ss, delay, u_sim, i)

    u_index = i - delay
    u = u_sim[:, u_index] if u_index >= 0 else np.zeros(model.nu)
    x_sim[:, i+1] = model.x_next(x_sim[:, i], u)

ax1.plot(range(T), u_sim[0], label="u1")
ax1.plot(range(T), u_sim[1], label="u2")
ax1.title.set_text("u (MPC)")
ax1.legend()

ax2.plot(range(T+1), x_sim[0], label="x1")
ax2.plot(range(T+1), x_sim[1], label="x2")
ax2.title.set_text("x (MPC)")
ax2.legend()

x_sim = np.zeros((model.nx, T+1))
u_sim = np.zeros((model.nu, T))
x_sim[:, 0] = [-11., 9]

deepc = DeePC_Controller(model, delay)

for i in range(T):
    u_sim[:, i] = deepc.step(model, x_sim[:, i], x_ss, u_ss, u_sim, model.y(x_sim), 0, [0], i)

    u_index = i - delay
    u = u_sim[:, u_index] if u_index >= 0 else np.zeros(model.nu)
    x_sim[:, i+1] = model.x_next(x_sim[:, i], u)

ax3.plot(range(T), u_sim[0], label="u1")
ax3.plot(range(T), u_sim[1], label="u2")
ax3.title.set_text("u (DeePC)")
ax3.legend()

ax4.plot(range(T+1), x_sim[0], label="x1")
ax4.plot(range(T+1), x_sim[1], label="x2")
ax4.title.set_text("x (DeePC)")
ax4.legend()

plt.show()

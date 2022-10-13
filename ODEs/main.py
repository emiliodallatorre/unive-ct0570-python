from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# Libraries for the simulation
import dopri as dopri
from scipy.integrate import solve_ivp


df = pd.DataFrame({'x_values': range(1, 101), 'y_values': np.random.randn(
    100)*15+range(1, 101), 'z_values': (np.random.randn(100)*15+range(1, 101))*2})


def racr_function(time, state):
    x, y, z = state

    dxdt = -0.04*x + 1e4*y*z
    dydt = 0.04*x - 1e4*y*z - 3e7*(y**2)
    dzdt = 3e7*(y**2)

    return dxdt, dydt, dzdt


# RK45
plt.subplot(221)
rk45_solution: np.array = solve_ivp(
    racr_function, [0, 500], (1, 0, 0), method="RK45")
plt.plot(rk45_solution.t, rk45_solution.y[0], label="x")
plt.plot(rk45_solution.t, rk45_solution.y[1], label="y")
plt.plot(rk45_solution.t, rk45_solution.y[2], label="z")

# Radau
plt.subplot(222)
radau_solution: np.array = solve_ivp(
    racr_function, [0, 500], (1, 0, 0), method="Radau")
plt.plot(radau_solution.t, radau_solution.y[0], label="x")
plt.plot(radau_solution.t, radau_solution.y[1], label="y")
plt.plot(radau_solution.t, radau_solution.y[2], label="z")

# LSODA
plt.subplot(223)
lsoda_solution: np.array = solve_ivp(
    racr_function, [0, 500], (1, 0, 0), method="LSODA")
plt.plot(lsoda_solution.t, lsoda_solution.y[0], label="x")
plt.plot(lsoda_solution.t, lsoda_solution.y[1], label="y")
plt.plot(lsoda_solution.t, lsoda_solution.y[2], label="z")

# BDF
plt.subplot(224)
bdf_solution: np.array = solve_ivp(
    racr_function, [0, 500], (1, 0, 0), method="BDF")
plt.plot(bdf_solution.t, bdf_solution.y[0], label="x")
plt.plot(bdf_solution.t, bdf_solution.y[1], label="y")
plt.plot(bdf_solution.t, bdf_solution.y[2], label="z")

# Show the graph
plt.savefig("out.png")

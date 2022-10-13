from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp


def benchmark_function(function, * args, ** kwargs):
    import time
    start = time.time()
    output = function(*args, **kwargs)
    end = time.time()

    return output, round((end - start) * 1000, 3)


def racr_function(time, state):
    x, y, z = state

    dxdt = -0.04*x + 1e4*y*z
    dydt = 0.04*x - 1e4*y*z - 3e7*(y**2)
    dzdt = 3e7*(y**2)

    return dxdt, dydt, dzdt


# RK45
plt.subplot(221)
rk45_solution, time = benchmark_function(
    solve_ivp, racr_function, [0, 500], (1, 0, 0), method="RK45")
plt.title(f"RK45 - {time} ms")
plt.plot(rk45_solution.t, rk45_solution.y[0], label="x")
plt.plot(rk45_solution.t, rk45_solution.y[1], label="y")
plt.plot(rk45_solution.t, rk45_solution.y[2], label="z")

# Radau
plt.subplot(222)
radau_solution, time = benchmark_function(
    solve_ivp, racr_function, [0, 500], (1, 0, 0), method="Radau")
plt.title(f"Radau - {time} ms")
plt.plot(radau_solution.t, radau_solution.y[0], label="x")
plt.plot(radau_solution.t, radau_solution.y[1], label="y")
plt.plot(radau_solution.t, radau_solution.y[2], label="z")

# LSODA
plt.subplot(223)
lsoda_solution, time = benchmark_function(
    solve_ivp, racr_function, [0, 500], (1, 0, 0), method="LSODA")
plt.title(f"LSODA - {time} ms")
plt.plot(lsoda_solution.t, lsoda_solution.y[0], label="x")
plt.plot(lsoda_solution.t, lsoda_solution.y[1], label="y")
plt.plot(lsoda_solution.t, lsoda_solution.y[2], label="z")

# BDF
plt.subplot(224)
bdf_solution, time = benchmark_function(
    solve_ivp, racr_function, [0, 500], (1, 0, 0), method="BDF")
plt.title(f"BDF - {time} ms")
plt.plot(bdf_solution.t, bdf_solution.y[0], label="x")
plt.plot(bdf_solution.t, bdf_solution.y[1], label="y")
plt.plot(bdf_solution.t, bdf_solution.y[2], label="z")

# Show the graph
plt.tight_layout()
plt.savefig("out.png")
plt.show()

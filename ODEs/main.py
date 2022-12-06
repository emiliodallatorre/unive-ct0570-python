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


for i, method in enumerate(["RK45", "Radau", "LSODA", "BDF"]):
    solution, time = benchmark_function(solve_ivp, racr_function, [0, 500], [1, 0, 0], method=method)

    plt.subplot(220 + i + 1)
    plt.title(f"{method} - {time} ms")
    plt.plot(solution.t, solution.y[0], label="x")
    plt.plot(solution.t, solution.y[1], label="y")
    plt.plot(solution.t, solution.y[2], label="z")

# Show the graph
plt.tight_layout()
plt.savefig("out.png")

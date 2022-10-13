from scipy.integrate import solve_ivp
from numpy import array, linspace
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_style("white")	
fig,ax=plt.subplots(3,1,figsize=(10,15), sharex=True)

def fun(t,x):
	p = -50
	return p*x,

if __name__ == "__main__":

	y0 = [1] 
	t0 = 0
	t_max = 1
	res_rk45  = solve_ivp(fun, [0, t_max], y0, method="RK45")
	res_rk23  = solve_ivp(fun, [0, t_max], y0, method="RK23")
	steps = 30
	y_em = 1
	res_euler = []
	stepsize=t_max/steps
	print(stepsize)
	for time in linspace(0,t_max,steps):
		res_euler.append(y_em)
		dy = fun(time, y_em)
		y_em += dy[0]*stepsize
	
	ax[2].plot(linspace(0,t_max,steps), res_euler, "k--", label="Euler")
	
	colors=[]
	for i, v in enumerate(res_rk45.t[:-1]):
		colors.append(v-res_rk45.t[i+1])
	colors.append(colors[-1])
	colors = array(colors)

	colors_23=[]
	for i, v in enumerate(res_rk23.t[:-1]):
		colors_23.append(v-res_rk23.t[i+1])
	colors_23.append(colors_23[-1])
	colors_23 = array(colors_23)

	# normal = mpl.colors.Normalize(vmin=min(min(colors), min(colors_23)), vmax=max(max(colors), max(colors_23)))

	ax[1].plot(res_rk45.t, res_rk45.y[0], "k--", label="RK45")
	ax[1].scatter(res_rk45.t, res_rk45.y[0], marker="s", c= colors, cmap="seismic")
	
	ax[0].plot(res_rk23.t, res_rk23.y[0], "k--", label="RK23")
	ax[0].scatter(res_rk23.t, res_rk23.y[0], marker="s", c= colors_23, cmap="seismic")
	
	ax[1].set_xlabel("Time")
	ax[0].legend()
	ax[1].legend()
	ax[2].legend()

	fig.tight_layout()
	fig.savefig("dopri.png")
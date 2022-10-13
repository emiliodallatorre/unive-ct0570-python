import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# prepare figures and style
sns.set_style("white")
fig, ax = plt.subplots(1,1,figsize=(7,7))

def deriv(x,v,omega):
	""" 
		Code for ODE derivative calculation: 
			dx/dt = v
			dv/dt = -omega^2 x 
		We return a tuple with the result of the two ODEs.
	"""
	return v, -omega**2 * x


if __name__ == '__main__':

	# parameters of the model
	v0 = 0
	x0 = 1
	omega = 1
	t_max = 30

	# test delta=0.1
	delta = 0.1
	steps = int(t_max/delta)
	vector_x = []
	# initial state
	v = v0; x = x0
	for time in np.linspace(0,t_max,steps):
		dx, dv = deriv(x, v, omega)
		v += dv * delta
		x += dx * delta
		vector_x.append(x)
	times = np.linspace(0,t_max,steps)
	ax.plot(times, np.cos(omega*times), color="black", linestyle="--", lw=2, label="Analytic solution")
	ax.plot(times, vector_x,"o-",ms=3, label="Euler's method, $\Delta=0.1$")
	


	ax.legend(loc="best")
	ax.set_xlabel("Time")
	ax.set_ylabel("x")
	fig.savefig("HO1.png")

	# plot the two simulations
	delta = 0.05
	steps = int(t_max/delta)
	vector_x = []
	# initial state
	v = v0; x = x0
	for time in np.linspace(0,t_max,steps):
		dx, dv = deriv(x, v, omega)
		v += dv * delta
		x += dx * delta
		vector_x.append(x)
	ax.plot(np.linspace(0,t_max,steps), vector_x,"o-",ms=3, label="Euler's method, $\Delta=0.05$")
	
	# plot the two simulations
	delta = 0.001
	steps = int(t_max/delta)
	vector_x = []
	# initial state
	v = v0; x = x0
	for time in np.linspace(0,t_max,steps):
		dx, dv = deriv(x, v, omega)
		v += dv * delta
		x += dx * delta
		vector_x.append(x)
	ax.plot(np.linspace(0,t_max,steps), vector_x,"o-",ms=3, label="Euler's method, $\Delta=0.01$")

	ax.plot(times, np.cos(omega*times+np.sin(np.pi)), color="black", linestyle="--", lw=2)
	

	ax.legend(loc="best")
	ax.set_xlabel("Time")
	ax.set_ylabel("x")
	fig.tight_layout()
	fig.savefig("HO.png")
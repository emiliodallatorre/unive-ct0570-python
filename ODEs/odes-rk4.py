import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# prepare figures and style
sns.set_style("white")
fig, ax = plt.subplots(1,1,figsize=(7,7))

def deriv(x,t,k1,k2):
	""" 
		Code for ODE derivative calculation: dG/dt = k1-k2*G. 
		Note that the derivative is only a function of the current state, 
		we are not using time at all.
	"""
	return k1-k2*x


if __name__ == '__main__':

	# parameters of the model
	par1 = 1e0
	par2 = 2e-2

	delta = 1

	# plot the derivatives
	for c in np.linspace(1e-1,1e2,15):
		for r in np.linspace(1e-1,1e2,15):
			res = deriv(r,c,par1,par2)
			ax.quiver(c, r, delta, res*delta, color="lightgray")

	# plot the two simulations
	h = 10
	g0 = 21
	t_max = 100
	steps = int(t_max//h)
	g = g0
	for time in np.linspace(0, t_max, steps):
		k1 = deriv(g, time, par1, par2)
		k2 = deriv(g+h/2*par1, time+h/2, par1, par2)
		k3 = deriv(g+h/2*par2, time+h/2, par1, par2)
		k4 = deriv(g+h*k3, time+h, par1, par2)
		g = g+1/6*(k1+2*k2+2*k3+k4)*h		
		ax.plot(time,g,"ro-",ms=1)
	
	g0 = 93
	g = g0
	for time in np.linspace(0, t_max, steps):
		k1 = deriv(g, time, par1, par2)
		k2 = deriv(g+h/2*par1, time+h/2, par1, par2)
		k3 = deriv(g+h/2*par2, time+h/2, par1, par2)
		k4 = deriv(g+h*k3, time+h, par1, par2)
		g = g+1/6*(k1+2*k2+2*k3+k4)*h	
		ax.plot(time,g,"bo-",ms=1)

	# custom legend and figure cosmetics stuff
	from matplotlib.lines import Line2D
	custom_lines = [	Line2D([0], [0], color="lightgray", lw=4),
                		Line2D([0], [0], linestyle=":", color="blue", lw=2),
                		Line2D([0], [0], linestyle=":", color="red", lw=2) ]
	ax.legend(custom_lines, ['Derivative', 'Starting from G=93', 'Starting from G=21'], loc="upper right")
	ax.set_xlabel("Time")
	ax.set_ylabel("Amount of G")
	fig.tight_layout()
	fig.savefig("fields-rk4-%.2f.png" % h)
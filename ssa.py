import numpy as np
import matplotlib.pyplot as plt 

class Model:

	def __init__(self, species=[], state=[], parameters=[], reactants_stoic=[], products_stoic=[]):
		""" 
			Initialization of the object with the information about the model. 
		"""
		self._species = species
		self._state = np.array(state)
		self._parameters = parameters
		self._reactants_stoic = np.array(reactants_stoic)
		self._products_stoic = np.array(products_stoic)
		self._propensities = np.zeros(len(self._parameters))
		self._time = 0

	def _get_h(self, index):
		""" 
			Calculate the combinations of reactants. 
		"""
		vector = self._reactants_stoic[index]
		if sum(vector)==0: return 1 # all zeros
		elif sum(vector)==1: # just one reactant
			return self._state[np.where(vector==1)]

		# to be continued..!
			
	def _update_propensities(self):
		""" 
			Update the propensities and a0 according to the current state.
		"""
		for m in range(len(self._parameters)):
			a = self._get_h(m) * self._parameters[m]
			self._propensities[m] = a
		self._a0 = sum(self._propensities)

	def SSA_step(self, tmax):
		""" 
			Performs a SSA step. Tmax is necessary in case the reactants are over
			and we must skip to the end of the simulation.
		"""
		self._update_propensities()
		if self._a0==0: return tmax, self._state.copy()
		tau = self._calculate_tau()
		index = self._get_next_reaction()
		self._time += tau
		change_vector = self._products_stoic[index] - self._reactants_stoic[index]
		self._state += change_vector
		return self._time, self._state.copy()

	def _calculate_tau(self):
		""" 
			Calculate the waiting time before the next reaction.
		"""
		return (1/self._a0) * (np.log(1/np.random.uniform()))

	def _get_next_reaction(self):
		""" 
			Calculate the next reaction with a roulette wheel.
		"""
		position = 0
		selected = 0
		rnd2 = self._a0*np.random.uniform()
		while (position<rnd2):
			position+=self._propensities[selected]
			selected+=1
		return selected-1

	def simulate(self, tmax):
		""" 
			Perform a complete SSA simulation up to tmax. Returns the time-series.
		"""
		states = []
		times  = []

		while self._time <= tmax:
			new_time, new_state = self.SSA_step(tmax)
			times.append(new_time)
			states.append(new_state)
			
		return np.array(times), np.array(states) 

if __name__ == "__main__":

	colors=["orange", "blue"]

	fig, ax = plt.subplots(1,1)

	for i in range(50):
		# Implementing the model:
		# A -> B
		# B -> A 
		M = Model(
			species=["S1", "S2"], 
			state = [100, 0],
			parameters = [0.5, 1e-2],
			reactants_stoic = [[1,0], [0,1]],  
			products_stoic = [[0,1], [1,0]]
		)
		times, states = M.simulate(tmax=10)
		for n, sp in enumerate(M._species):
			ax.plot(times, states.T[n], alpha=0.1, color=colors[n])
	ax.set_xlabel("Time [a.u.]")
	ax.set_ylabel("Molecules [#]")
	fig.tight_layout()
	fig.savefig("prova.png")
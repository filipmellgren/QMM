# Created by Filip Mellgren, filip.mellgren@su.se
# Functions related to value function iteration used for course quantitative macroeconomic methods
import numpy as np
from numba import jit, njit, prange
from numba.typed import Dict
from scipy import optimize
import math

@jit(nopython=True, fastmath=True, parallel = True)
def value_function_iterate(V, transition_matrix, reward_matrix, stoch_states, det_states,  disc_factor, tol = 1e-6):
	""" Value function iterate over two state variables, one stochastic, one deterministic

	Value function iterate over a value function with two state variables, one stochastic and one deterministic.
	Outer loop: iterates over stochastic dimension. This dimension tends to be small and the expected value required in the deterministic dimension can be used and over again.
	Inner loop: iterates over the deterministic dimension and finds highest rewarding action given both stochastic and deterministic state.

	Highest state is found by linear search over the possible actions. 
	TODO: Future versions might find the optimal action using smarter methods.

	Parameters
	----------
	V_guess : numpy array dim(stoch_states) x dim(det_states). An initial guess for the value function matrix
	transition_matrix : numpy array of probabilities of going from row to col
	reward_matrix : numpy array a 3d matrix containing the direct reward of each action for each stochastic X deterministic state
	stoch_states : numpy array containing all stochastic states
	det_states : numpy array containing all deterministic states
	disc_factor : a discount factor 0 <= disc_factor < 1. Won't convert if >= 1 (Blackwell's conditions)
	tol : optional, tolerance level below which the function is considered converged

	Returns
	-------
	V
	  A value function indexed by stochastic state dimension x deterministic state dimension
	POL
		Numpy array of dimensions stoch_states by det_states with index of optimal action
	"""
		
		#assert	disc_factor < 1, "Discount factor has to be less than unity for convergence"
	#V = V_guess.astype("float16")

	V_new = np.copy(V)
	POL = np.zeros(V_new.shape, dtype=np.int16)
	diff = 100.0
	
	while diff > tol:
		diff = 0.0

		for s_ix in range(stoch_states.shape[0]):
			P = transition_matrix[s_ix, :]
			V = np.copy(V_new)
			E_action_val = np.dot(P, V) # Vector valued. One value per action.
			for d_ix in prange(det_states.shape[0]):
				v = V[s_ix, d_ix]
				V_new_cands = reward_matrix[s_ix, d_ix,:] + disc_factor * E_action_val
				pol = np.argmax(V_new_cands)
				POL[s_ix, d_ix] = pol
				v_new = V_new_cands[pol]
				V_new[s_ix, d_ix] = v_new
				diff = max(diff, abs(v - v_new))

	return(V, POL)
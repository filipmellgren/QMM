# Created by Filip Mellgren, filip.mellgren@su.se
# Functions related to value function iteration used for course quantitative macroeconomic methods
import numpy as np
from numba import jit, njit, prange
from numba.typed import Dict
from scipy import optimize
from src.reward_funcs import find_asset_grid
import ipdb
from interpolation import interp
from src.reward_funcs import get_util

import math
#import ipdb
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
	assert np.all(np.sum(transition_matrix, 1) == 1), "Rows in transition matrix don't sum to 1"
	V_new = np.copy(V)
	POL = np.zeros(V_new.shape, dtype=np.int16)
	diff = 100.0
	
	while diff > tol:
		diff = 0.0

		for s_ix in range(stoch_states.shape[0]):
			P = transition_matrix[s_ix, :]
			V = np.copy(V_new)
			E_action_val = np.dot(V, P.T) # Vector valued. One value per action.
			for d_ix in prange(det_states.shape[0]):
				v = V[d_ix, s_ix]
				V_new_cands = reward_matrix[s_ix, d_ix,:] + disc_factor * E_action_val
				pol = np.argmax(V_new_cands)
				POL[d_ix, s_ix] = pol
				v_new = V_new_cands[pol]
				V_new[d_ix, s_ix] = v_new
				diff = max(diff, abs(v - v_new))

	return(V, POL)

def egm(V, transition_matrix, action_states, asset_states, rate, params, tol = 1e-6):
	""" Implements the Endogenous Grid Method for solving the household problem.
	
	# TODO: only works for infinitely lived household problems. 

	Useful reference: https://alisdairmckay.com/Notes/HetAgents/EGM.html
	The EGM method solves the household problem faster than VFI by using the HH first order condition to find the policy functions. Attributed to Carroll (2006).

	Roughly the algorithm for updating the diff between policy guess and updated policy is:
		* Calculate future consumption using budget constraint and policy guess
		* Calculate marginal utility of consumption today using the Euler Equation. This requires the expected marginal utility of consumption tomorrow. Solve for consumption today. Easy with CRRA utility (which implementation assumes, not with parameter 1 though, the log utility case, approximate with value close to 1).
		* Above gives mapping from action to assets today. Want to flip this relation, do that using interpolation and evaluate at the exogenous grid. Extrapolate using borrowing constraint and maximum possible asset value (later shouldn't be a problem ideally, just increase upper bound).
		* Update policy and calculate the distance metric, compare and terminate or reiterate.

	Parameters
	----------
	TODO: update the below.
	V : numpy array dim(stoch_states) x dim(det_states). An initial guess for the value function matrix
	transition_matrix : numpy array of probabilities of going from row to col
	reward_matrix : numpy array a 3d matrix containing the direct reward of each action for each stochastic X deterministic state
	stoch_states : numpy array containing all stochastic states
	det_states : numpy array containing all deterministic states
	tol : optional, tolerance level below which the function is considered converged

	Returns
	-------
	V
	  A value function indexed by stochastic state dimension x deterministic state dimension
	POL
		Numpy array of dimensions stoch_states by det_states with index of optimal action
	"""
	
	income_states = params["income_states"]
	action_n = action_states.shape[0]
	income_n = income_states.shape[0]
	action_states = np.tile(action_states.T, income_n).reshape(income_n, action_n).T
	exog_grid = np.copy(action_states)
	policy_guess = np.full((action_n, income_n), np.min(action_states))
	policy_guess += np.tile(np.linspace(0.001, 0.01, action_n).T, income_n).reshape(income_n, action_n).T

	P = transition_matrix

	diff = tol + 1
	while diff > tol:
		diff = 0

		mu_cons_fut = ((1+rate) * action_states + income_states - policy_guess)**(-params["risk_aver"])
		Ecf = np.matmul(mu_cons_fut, P.T)
		cons_today = (params["disc_fact"] * (1+rate) * Ecf)**(-1/params["risk_aver"])
		endog_assets = 1/(1+rate) * (cons_today + action_states - income_states) # Mapping from action to assets. Want mapping from assets to action. 
		policy_guess_upd = np.empty(policy_guess.shape)
		for s in range(income_n):
			# Invert the mapping. Extrapolate outside range using borrowing constraint and max assets in exogenous grid. 
			exog_action = action_states[:,s]
			policy_guess_upd[:,s] = np.interp(x = exog_grid[:,s], xp = endog_assets[:,s], fp = exog_action, left = exog_action[0], right = exog_action[-1]) 
			
		diff = max(np.max(np.abs(policy_guess_upd - policy_guess)), diff)
		policy_guess = np.copy(policy_guess_upd)

		reward = (np.power(cons_today, 1-params["risk_aver"]))/(1-params["risk_aver"])
		reward[cons_today<0] = -np.inf 
		V_upd = reward + params["disc_fact"] * V
		V = np.copy(V_upd)
	return(V, policy_guess)
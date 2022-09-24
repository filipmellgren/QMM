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
		
		#assert	disc_factor < 1, "Discount factor has to be less than unity for convergence"
	#V = V_guess.astype("float16")
	assert np.all(np.sum(transition_matrix, 1) == 1), "Rows in transition matrix don't sum to 1"
	P = transition_matrix[0, :]
	assert (np.dot(P, V)).shape[0] == reward_matrix.shape[2], "The size of the expectation of an action and the reward of that action are not the same!"
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


def egm(V, transition_matrix, action_states, asset_states, r, params, tol = 1e-6):
	
	income_states = params["income_states"]
	action_n = action_states.shape[0]
	income_n = income_states.shape[0]
	action_states = np.tile(action_states.T, 2).reshape(income_n, action_n).T
	asset_states = np.tile(asset_states.T, 2).reshape(income_n, action_n).T
	
	policy_guess = np.full((action_n, income_n), -3.0)
	V_guess = np.empty((action_n, income_n))
	P = transition_matrix
	diff = tol + 1
	while diff > tol:
		cons_fut = ((1+r) * action_states + income_states - policy_guess)**(-params["risk_aver"])

		Ecf = np.matmul(cons_fut, P) # TODO double check with einstein summation

		cons_today = (params["disc_fact"] * (1+r) * Ecf)**(-1/params["risk_aver"]) # From Euler Equation
		endog_assets = 1/(1+r) * (cons_today + action_states - income_states)# From Budget Constraint
		# TODO: Do we need to check kink somewhere?
		
		policy_guess_upd = np.empty(policy_guess.shape)
		for s in range(income_n):
			policy_guess_upd[:,s] = interp(action_states[:,s], asset_states[:,s],endog_assets[:,s])
		
		diff = np.max(abs(policy_guess_upd - policy_guess))
		policy_guess = policy_guess_upd

		reward = (np.power(cons_today, 1-params["risk_aver"]))/(1-params["risk_aver"])
		reward[cons_today<0] = -np.inf 
		V_upd = reward + params["disc_fact"] * V
		V = np.copy(V_upd)

	return(V, policy_guess)

def end_grid_method(V, rate, borrow_constr, action_states, consumption_matrix, params, tol = 1e-6):
	""" Implements the Endogenous Grid Method for solving the household problem.
	
	The EGM method solves the household problem faster than VFI by using rthe HH first order condition to find the policy functions. Attributed to Carroll (2006).

	TODO: need a new asset states in each iterration
	Parameters
	----------
	V : numpy array dim(stoch_states) x dim(det_states). An initial guess for the value function matrix
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
	# TODO: improve this function so it works for any given input. Extremely hacky atm.
	# https://alisdairmckay.com/Notes/HetAgents/EGM.html
	
	# Transpose V so it conforms with slides:
	V = V.T # V is 1000 by 3
	income_states = params["income_states"]
	action_n = action_states.shape[0]
	income_n = income_states.shape[0] 
	action_states = np.tile(action_states.T, 2).reshape(income_n, action_n).T
	borrow_constr = borrow_constr[0] # TODO: double check this. DO we use this period or next period's orrow constraint? 
	assert(V.shape == (action_n, income_n))

	consumption_matrix = np.amax(consumption_matrix, axis = 2) # TODO: is this correct? We only care about max consumption st bc?
	# TODO: implement BC somewhere here
	
	V_d = np.tile(np.linspace(0.01, 10.0, num=action_n), income_n).reshape(income_n, action_n).T# dV_da(V, action_states, income_states, borrow_constr, params) +0.1 # TODO: note some values in the matrix are negative. Guess marginal utility here is infinite? TODO: this is a cube, not a matrix. What dimension should I use?
	#V_d = V_d.T
	P = params["transition_matrix"]
	
	EV = np.empty((action_n, income_n))
	EV_d = np.empty((action_n, income_n))
	
	diff = tol + 1
	while diff > tol:
		#for s_next in range(income_n):
		#	EV += np.matmul(V, P[:,s_next])
		#	EV_d += np.matmul(V_d, P[:,s_next])
		for s in range(income_n):
			EV[:,s] = np.matmul(V, P[s, :])
			EV_d[:,s] = np.matmul(V_d, P[s, :])
		
		# Ideal consumption from FOC
		consumption = params["disc_fact"] * EV_d**(-1/params["risk_aver"]) # 1000 by 2. consumption(a', s) = c(c(a'|s) + a', s)

		# Implied assets from budget constraint
		implied_assets = (consumption + action_states - income_states)/(1+rate)

		# Identify what assets are not consistent with the budget constraint
		violated_budget = implied_assets > consumption - income_states - borrow_constr

		# Where the budget constraint is violated, we must force consumption down to the highest value where constraint still holds
		violated_bc_cons = (consumption + action_states - borrow_constr) * violated_budget  # TODO double check this line. Understood as assets(1+r) + income - borrow_consraint

		consumption = consumption * (np.logical_not(violated_budget)) + violated_budget * violated_bc_cons
				
		V_upd = consumption**(1-params["risk_aver"]) / (1 - params["risk_aver"]) + params["disc_fact"] * EV # TODO: Not sure about correct dimension here.
		# TODO: interpolate the consumption policy and value function onto the cash on hand grid.
		# TODO: is above even necessary??
		V_d = dV_da(consumption, rate, params) 
		diff = np.max(abs(V_upd - V))
		V = V_upd
	
	assert(EV.shape == (action_n, income_n))
	
	return(V)

def dV_da(consumption, rate, params):
	""" Compute derivative wrt deterministic argument of V. 

	Uses the envelope theorem to compute the derivative of V.
	"""
	# TODO: this is not a matrix, it is a 3d tensor 
	#action_spacing = np.diff(action_states, prepend = borrow_constr-0.00264519) # TODO: don't hard code
	#stoch_spacing = 
	#V_d = np.gradient(V, action_spacing, axis = 0)
	V_d = consumption**(-params["risk_aver"]) * (1+rate) # du/dc * dc/da
	#assert(np.all(V_d) > 0)
	return(V_d)



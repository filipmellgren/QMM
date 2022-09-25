# Solve for the ergodic distribution
import numpy as np
from numba import jit, njit, prange
import ipdb
#@jit(nopython=True, parallel = True)
def iterate_distr(distr_guess, policy, action_set, transition_matrix, inc_size, asset_size, tol):
	# TODO: make sure that the policy grid conforms with the action set grid. 
	diff = 100.0
	while diff > tol:
		diff = 0.0
			# Transition vector is the transition probability to a state (col)  from all possible states in the rows
		for sl in range(inc_size):
			transition_vector = transition_matrix[:,sl] # From sl to sj # TODO: correct index?
			for ak in range(asset_size):
				indic = np.abs(action_set[ak] - policy) < 0.000001 # TODO make these two compatible
				d_new = np.sum(np.dot((indic * distr_guess), transition_vector))
				d_prev = distr_guess[ak, sl]
				diff = max(diff, abs(d_new - d_prev))
				distr_guess[ak, sl] = d_new
		distr_guess = distr_guess/np.sum(distr_guess) # TODO: this is super hacky! It is needed bc I think the distribution starts to drift away from 1. 
	return(distr_guess)

def solve_distr(policy, action_set, params):
	'''
	Given interest rate, houesehold policies, and exogenous income process, what distribution do we end up with?
	'''
	distr_guess = np.full((
		params["action_grid_size"],
		params["inc_grid_size"]), 1/(params["inc_grid_size"]*params["action_grid_size"]))

	assert np.isclose(np.sum(distr_guess), 1), "Initial distribution guess doesn't sum to 1"

	distr_upd = np.empty(distr_guess.shape)
	transition_matrix = params["transition_matrix"]
	tol = params["distr_tol"]
	diff = 100.0
	i = 0
	
	row_sums = np.sum(transition_matrix, axis = 1)
	assert np.all(np.isclose(row_sums, 1)), "Transition matrix is not right stochastic, given row, columns should sum to one."

	assert policy.shape == (params["action_grid_size"], params["inc_grid_size"]), "Policy index not of expected dimension. It is: " + str(policy.shape)

	distr_guess = iterate_distr(distr_guess, policy, action_set, transition_matrix, params["inc_grid_size"], params["action_grid_size"], tol)

	sum_distribution = np.sum(distr_guess)

	assert abs(sum_distribution - 1)<0.001, "Distribution sums to " + str(sum_distribution)
	assert np.all(distr_guess >= 0), "Negative values in distribution"
	return(distr_guess)

def check_mc(params, policy, ergodic_distr):
	'''
	Check Market clearing
	Need mass of people in each asset state
	Multiply mass with asset demand for people in that state
	Aggregate demand and compare with net assets
	'''
	asset_demand = np.sum(ergodic_distr * policy)
	diff = asset_demand - params["asset_net"] 
	return(diff)

def policy_ix_to_policy(policy_ix, stoch_states, det_states, action_set):
	''' Convert a policy index matrix to a policy matrix

	A policy index matrix with rows as stochastic variable, and cols as determinstic variable is converted to a sma shaped policy matrix, with the actual optimal action for each state.
	'''

	policy = np.zeros(policy_ix.shape)
	for row in range(stoch_states.shape[0]):
		for col in range(det_states.shape[0]):
			policy[row, col] = action_set[policy_ix[row, col]]
	return(policy)
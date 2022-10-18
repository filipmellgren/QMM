# Solve for the ergodic distribution
import numpy as np
from numba import jit, njit, prange
import ipdb
#@jit(nopython=True, parallel = True)
def iterate_distr(distr_guess, policy, action_set, transition_matrix, inc_size, asset_size, tol):
	diff = 100.0
	while diff > tol:
		diff = 0.0
			# Transition vector is the transition probability to a state (col)  from all possible states in the rows
		for sl in range(inc_size):
			P_vec = transition_matrix[:,sl] 
			for ak in range(asset_size):
				indic = np.abs(action_set[ak] - policy) < 0.000001 
				d_new = np.sum(np.dot((indic * distr_guess), P_vec))
				d_prev = distr_guess[ak, sl]
				diff = max(diff, abs(d_new - d_prev))
				distr_guess[ak, sl] = d_new
		#distr_guess = distr_guess/np.sum(distr_guess) # TODO: this is super hacky! It is needed bc I think the distribution starts to drift away from 1.
			try:
				assert(np.isclose(np.sum(distr_guess), 1)), print(np.sum(distr_guess))
			except AssertionError as e:
				ipdb.set_trace()
				print(np.sum(distr_guess))
				raise e
	return(distr_guess)

@jit(nopython=True, parallel = True)
def kieran_finder(distr_guess, policy, action_set, transition_matrix, inc_size, asset_size, tol):
	# Based on Kieran's matlab code, the slow version.
	lambd = np.copy(distr_guess)
	diff = 100.0
	while diff > tol:
		diff = 0.0
		lambdap = np.full(lambd.shape, 0.0)
		for l in range(inc_size):
			for j in range(inc_size):
				prob = transition_matrix[j,l]
				for k in prange(asset_size):
					indic = policy[:,j] == action_set[k]
					lambdap[k,l] = lambdap[k,l] + prob * np.sum(lambd[indic,j])

		dlambda = np.max(np.abs(lambdap - lambd))
		diff = max(dlambda, diff)
		lambd = np.copy(lambdap)
		# Below assertion cannot be used in JIT mode
	#assert(abs(np.sum(distr_guess) - 1)<0.00001), print(np.sum(distr_guess))
	return(lambd)

def solve_distr(policy, action_set, params):
	'''
	Given interest rate, houesehold policies, and exogenous income process, what distribution do we end up with?
	'''
	distr_guess = np.full((
		params["action_grid_size"],
		params["inc_grid_size"]), 1/(params["inc_grid_size"]*params["action_grid_size"]))
	transition_matrix = params["transition_matrix"]

	# Check that input is fine:
	row_sums = np.sum(transition_matrix, axis = 1)
	assert np.all(np.isclose(row_sums, 1)), "Transition matrix is not right stochastic, given row, rows should sum to one."
	assert policy.shape == (params["action_grid_size"], params["inc_grid_size"]), "Policy index not of expected dimension. It is: " + str(policy.shape)
	assert np.isclose(np.sum(distr_guess), 1), "Initial distribution guess doesn't sum to 1"

	# Main function:

	distr_guess = kieran_finder(distr_guess, policy, action_set, transition_matrix, params["inc_grid_size"], params["action_grid_size"], params["distr_tol"])
	#distr_guess = iterate_distr(distr_guess, policy, action_set, transition_matrix, params["inc_grid_size"], params["action_grid_size"], params["distr_tol"])

	# Check that output is fine:
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
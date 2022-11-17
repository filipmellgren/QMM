import numpy as np
import time

# Solve for nonlinear steqady state. DOne!


# Solve for transition path to small shock (e.g. TFP) imposing market clearing
def solve_trans_path(ss):

	# Choose time T when ecobnomy has returned to steady state
	T = 100
	#  Guess path for capital and labor
		# K can be a linear decay
	K = ss.K 
	diff = 100
	tol = 1e-2
	# Weight must be tiny due to extreme sensitivity
	weight = 1e-4
	K_guess = np.repeat(K, T+1) # TODO: should probably be a linear decay/increase instead
	start = time.time()
	while diff > tol:
		end = time.time()
		print("Time elapsed:")
		print(end - start)
		start = time.time()
		# Solve the value function/household policy backwards
		policy = backward_iterate_policy(K_guess, policy_ss, T, ss)
		# From SS, solve forward using policy functions and idiosyncratic Markov process
		distr = forward_iterate_distribution(policy, T, ss)
		# Calculate capital supplied at each point in time
		flat_pols = []
		for pol in policy:
			flat_pols.append(pol.flatten())
		K_HH = np.sum(np.asarray(flat_pols) * np.asarray(distr), axis = 1)

		# Check max difference
		diff = np.max(np.abs(K_HH - K_guess))
		print(diff)
		# Update guess for K
		K_guess = (1- weight) * K_guess + (weight) * K_HH
	return(K_guess)


# Use the Impulse Response Function as a numerically computed derivative

# Treat the value of a variable at point t as the sum of responses to all past shocks

def backward_iterate_policy(K_guess, policy_ss, T, ss):
	''' Iterate backwards to get policies in transition.
	ss is an instance of Market
	'''
	policy_list = [policy_ss]
	for time in reversed(range(T)):
		K_t = K_guess[time]
		ss.set_capital(K_t) # OBS: this updates the market object's rate (so it is no longer steady state)
		policy_prev = policy_list[-1]
		policy_t = egm_update(policy_prev, ss.P, ss.r, ss.wage, ss.tax, ss.L_tilde, ss.mu, ss.gamma, ss.beta, ss.delta, ss.state_grid, ss.asset_states) # Can use egm_update, do not iterate on it
		policy_list.append(policy_t)
	policy = np.asarray(policy_list)
	return(policy)

def forward_iterate_distribution(policy, T, ss):
	''' Forward iterate to find ergodic distributions
	This function is kind of slow compared to the backward iteration
	'''
	
	distr_list = []
	for time in range(T+1):
		policy_t = policy[time]
		policy_t_ix = value_array_to_index(policy_t.flatten(), ss.asset_states)
		P = get_transition_matrix(ss.Q, policy_t_ix, ss.state_grid) # TODO: is Q not changing? No, but P is, and that updates
		distr_t = get_distribution(P)
		distr_list.append(distr_t)
	distr = np.asarray(distr_list)
	return(distr)

ss = steady_state
tmp = solve_trans_path(ss)

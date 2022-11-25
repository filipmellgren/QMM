import numpy as np
from household_problem import egm_update, policy_to_grid
from distribution_funcs import get_transition_matrix, get_distribution, get_distribution_fast
import numba as nb
# Solve for nonlinear steqady state. DOne!
import ipdb

# Solve for transition path to small shock (e.g. TFP) imposing market clearing
def solve_trans_path(ss, T, distr0, policy_ss, tfp, K_guess):
	''' BKM algorithm for solving for transition path following shock in TFP.

	'''
	K0 = ss.K 
	loss = 100
	tol = 1e-4
	weight = 1e-1
		
	n_iters = 0

	K_HH_list = []

	while loss > tol and n_iters < 200:
		n_iters += 1

		policy, rate_path, wage_path = backward_iterate_policy(tfp, K_guess, policy_ss, T, ss)
		
		distr = forward_iterate_distribution(policy, distr0, T, ss)
		
		K_HH = K_supply(policy, distr, T)
		K_HH_list.append(K_HH) 

		# Check loss function
		diff_array = K_HH - K_guess[1:T+1]
		loss = np.max(diff_array)**2
				
		# Update capital demand guess
		K_guess = (1- weight) * K_guess[1:T+1] + (weight) * K_HH
		K_guess = np.insert(K_guess, 0, K0)

		weight = np.maximum(weight*1, 1e-5)

	return(K_guess, K_HH_list, tfp, T, rate_path, wage_path)

def K_supply(policy, distr, T):
	''' Find capital supplu from policy and distritbution of households
	
	Ensure distribution and policy are setup in a way such that the right mass coprresponds to the right policy.

	INPUT
	policy : is a numpy array of dimensions number of states not related to assets X number of possible asset states X T. It is flattened to work with the array. Interpret as one policy for each state in each period.
	distr : is a numpy array of dimensions total number of states X T. 

	OUTPUT
	K_HH : A numpy array of capital supplied for each t in T.

	'''
	flat_pols = []
	for pol in policy:
		flat_pols.append(pol.flatten())

	K_HH = np.sum(np.asarray(flat_pols) * np.asarray(distr), axis = 1)
	K_HH = K_HH[0:T] # HH savings in period t that should correspond to firm assets in period t+1
	return(K_HH)


def backward_iterate_policy(tfp, K_guess, policy_ss, T, ss):
	''' Iterate backwards to get policies in transition.
	ss is an instance of Market
	policy_ss : is the policy in steady state. A numpy array of income states X asset states containing values (not indices) of savings. 
	'''
	policy_list = [policy_ss]
	rate_fut = ss.r 
	wage_fut = ss.wage
	rate_path = [rate_fut]
	wage_path = [wage_fut]
	for time in reversed(range(T)):
		
		tfp_t = tfp[time] # First 'time' is T-1
		K_t = K_guess[time]
		ss.K = K_t # K is endogenous (from guess). Need to find fixed point st markets clear.
		ss.set_tfp(tfp_t, K_t) # Set A exogenously
		rate = ss.r
		wage = ss.wage
		
		policy_prev = policy_list[-1]

		policy_t = egm_update(policy_prev, ss.P, rate, rate_fut, wage, wage_fut, ss.tax, ss.L_tilde, ss.mu, ss.gamma, ss.beta, ss.delta, ss.state_grid, ss.asset_states) # Can use egm_update, do not iterate on it

		policy_list.append(policy_t)
		rate_fut = np.copy(rate)
		wage_fut = np.copy(wage)
		rate_path.append(rate_fut)
		wage_path.append(wage_fut)
	
	policy_list.reverse()
	rate_path.reverse()
	wage_path.reverse()
	policy = np.asarray(policy_list)
	
	return(policy, rate_path, wage_path)


def forward_iterate_distribution(policy, distr0, T, ss):
	''' Forward iterate to find ergodic distributions
	This function is kind of slow compared to the backward iteration.
	distr_guess : a guess for the distribution. A good guess is the distribution from a previous iteration. Is a list of length T.
	'''
	
	distr_list = [distr0]
	for time in range(T):
		policy_t = policy[time]
		#policy_t_ix = value_array_to_index(policy_t.flatten(), ss.asset_states)
		policy_t_ix, alpha_list = policy_to_grid(policy_t.flatten(), ss.asset_states)
		P = get_transition_matrix(ss.Q, nb.typed.List(policy_t_ix), nb.typed.List(alpha_list), ss.state_grid) 
		P = np.asarray(P)
		distr_t = get_distribution_fast(distr_list[-1], P, ss.state_grid, tol = 1e-12)
		#distr_t = get_distribution(P)
	
		distr_list.append(distr_t)
	distr = np.asarray(distr_list)
	return(distr)




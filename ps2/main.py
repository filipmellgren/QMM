# Problem set 2, money in the Huggett model
# Author: filip.mellgren@su.se

# LONG TERM ####
# TODO: implement the endogenous grid method
# TODO: check accuracy of the solution
# TODO: accelerator methods

import numpy as np
import scipy.stats as st
import scipy as sp
from tauchenhussey import tauchenhussey
import itertools
import ipdb
from numba import jit, njit
from numba.typed import Dict
import warnings

PARAMS = {
	"disc_fact": 0.993362,
	"risk_aver": 3,
	"borrow_c": -3,
	"asset_net": 0,
	"asset_max": 24,
	"ar1": 0.95, # rho
	"shock_mean": 0,
	"shock_var": 0.015,
	"value_func_tol": 1e-6,
	"distr_tol": 1e-8,
	"mc_tol": 1e-3,
	"inc_grid_size": 3, 
	"action_grid_size": 100, # TODO: set to 1000
}


income_states, transition_matrix = tauchenhussey(
	PARAMS["inc_grid_size"], # number of nodes for Z
	PARAMS["shock_mean"], # unconditional mean of process
	PARAMS["ar1"], # rho
	np.sqrt(PARAMS["shock_var"]), # std. dev. of epsilons
	np.sqrt(PARAMS["shock_var"])) # std. dev. used to calculate Gaussian # TODO: update this based on Floden rule I think
# TODO 0.051
PARAMS["income_states"] = np.copy(income_states[0])
PARAMS["asset_states"] = np.logspace(start = 0, stop = np.log10(PARAMS['asset_max'] - PARAMS['borrow_c']+1), num = PARAMS['action_grid_size']) + PARAMS['borrow_c'] -1
PARAMS["action_set"] = np.copy(PARAMS["asset_states"])
PARAMS["transition_matrix"] = transition_matrix

def floden_weight(ar1):
	'''
	Implements Flodén weighting
	'''
	weight = 0.5 + ar1/4
	return(weight)

def get_util(consumption, params):
	'''
	Calculate agent dirtect utility. 
	TODO: implement option to specify functional form
	'''
	
	utility = (consumption**(1-params["risk_aver"])-1)/(1-params["risk_aver"])
	utility[consumption<0] = -np.inf # TODO: minus infinity

	return(utility)

def calc_consumption(params, rate):
	income_states = params["income_states"]
	asset_states = params["asset_states"]
	action_set = params["action_set"]
	income, asset, saving = np.meshgrid(income_states, asset_states, action_set, sparse = True, indexing='ij')
	consumption = income + asset * rate - saving
	return(consumption)

def solve_hh(interest_rate, reward_matrix, params):
	# HH maximize. Considers for all states what the optimal action is and updates their policies based on this.
	# TODO: HH can be a class with V, policy as attributes
	# TODO: Implement Floden weight for tauchenhussey

	#state_space = np.array([x for x in itertools.product(income_states, asset_states)])

	#V = np.full((np.product(state_space.shape),action_set.shape[0]), 0)
	# Matrix dimension/depth is income, rows are asset dimension, columns are action dimension
	
	income_states = params["income_states"]
	asset_states = params["asset_states"]
	action_set = params["action_set"]

	V = np.full((
		params["inc_grid_size"],
		params["action_grid_size"], 
		params["action_grid_size"]), 1.0)
	
	V_new = np.copy(V)
	
	arbitrary_slice_index = 0
	#state_space = np.copy(V[:,:,arbitrary_slice_index])
	#ipdb.set_trace() # HERE
	V = value_function_iterate(V, transition_matrix, reward_matrix, params["income_states"], params["asset_states"], params["action_set"], params["disc_fact"], params["value_func_tol"])
	return(V)
	ipdb.set_trace()
	iteration = 0
	#ipdb.set_trace()
	it = np.nditer(V, flags=['multi_index'])
	diff = 100
	while diff > params["value_func_tol"]:
		iteration = iteration + 1
		diff = 0.0
		#ipdb.set_trace()
		for s in it:
			exp_val_next = params["transition_matrix"][it.multi_index[0]] @ np.amax(V, 2) # function of action

			value_now = np.amax(reward_matrix[it.multi_index] + params["disc_fact"] * exp_val_next)

			V_new[it.multi_index] = value_now
			diff = max(diff, np.abs(V_new[it.multi_index]- V[it.multi_index]))
			V = np.copy(V_new)
	return(V)

def solve_distr(policy, params):
	'''
	Given interest rate, houesehold policies, and exogenous income process, what distribution do we end up with?
	'''
	
	distr_guess = np.full((
		params["action_grid_size"],
		params["inc_grid_size"]), 1/(params["inc_grid_size"]*params["action_grid_size"]))

	action_dim = 0
	income_dim = 1
	#distr_guess = initial_lambda
	converged = False
	distr_upd = np.empty(distr_guess.shape)
	it = np.nditer(distr_guess, flags=['multi_index'])
	while not converged:
		for v in it:
			#ipdb.set_trace()
			
			# Transition vector is the transition probability to a state (col)  from all possible states in the rows
			transition_vector = params["transition_matrix"][:,it.multi_index[income_dim]] # 3 by 1
	
			a_k = params["action_set"][it.multi_index[action_dim]] # 1 by 1
			
			indicator = a_k == policy[it.multi_index[income_dim],:] # 1000 by 1
	
			distr_upd[it.multi_index] = indicator.T @ distr_guess @ transition_vector # 1 by 1
		converged = np.max(distr_upd - distr_guess) < params["distr_tol"]
		distr_guess = np.copy(distr_upd)
	
	sum_distribution = np.sum(distr_guess)
	#ipdb.set_trace()
	assert abs(sum_distribution - 1)<0.0001, "Distribution sums to " + str(sum_distribution)
	assert np.all(distr_guess >= 0), "Negative values in distribution"

	return(distr_upd)
 # parallel = True, fastmath = True
@jit(nopython=True, fastmath=True) # See https://numba.pydata.org/numba-doc/latest/user/parallel.html # TODO: make of type prange()
def value_function_iterate(V, transition_matrix, reward_matrix, income_states, asset_states, actions,  disc_factor, tol):
	#ipdb.set_trace()
	V_new = np.copy(V)
	POL = np.zeros(V_new[:,:,0].shape)
	diff = 100.0
	iteration = 0
	while diff > tol:
		iteration += 1			
		inc_ix = -1
		diff = 0.0
		for i in income_states:
			inc_ix += 1
			ass_ix = -1
			for a in asset_states:
				ass_ix += 1
				exp_val = np.zeros(actions.shape)
				v = np.max(V[inc_ix, ass_ix,:])
				act_ix = -1
				for act in actions: # Find expected value of each action
					act_ix += 1
					inc_ix2 = -1
					for i2 in income_states:
						inc_ix2 += 1
						exp_val[act_ix] += transition_matrix[inc_ix, inc_ix2] * np.max(V[inc_ix2, act_ix, :])
				v_new = np.max(reward_matrix[inc_ix, ass_ix,:] + disc_factor * exp_val) # 1 by 
				pol = np.argmax(reward_matrix[inc_ix, ass_ix,:] + disc_factor * exp_val)
				POL[inc_ix, ass_ix] = pol
				#if v_new < -10000:
		#		ipdb.set_trace()
				#	ipdb.set_trace()
				#	pass
				
				#V_new[inc_ix, ass_ix,act_ix] = v_new
				V_new[inc_ix, ass_ix,:] = v_new
				V = np.copy(V_new)
				diff = max(diff, abs(v - v_new))
			#	print(diff)
				#print(diff)
		#	diff = np.max(diff, np.abs(V_new[inc_ix, ass_ix] - V[inc_ix, ass_ix]))
				
	return(V, POL)

def check_mc(params, policy, ergodic_distr):
	'''
	Check Market clearing
	Need mass of people in each asset state
	Multiply mass with asset demand for people in that state
	Aggregate demand and compare with net assets
	'''
	#ipdb.set_trace()
	asset_demand = np.sum(ergodic_distr * policy)
	diff = asset_demand - params["asset_net"] 
	return(diff)

def solve_model(rate_guess, params):
	#params = PARAMS
	reward_matrix = get_util(calc_consumption(params, rate_guess), params)
	#ipdb.set_trace()
	V, policy_ix = solve_hh(rate_guess, reward_matrix, params) # Given rate, what would HH do? (all households behave same way, but end up in different states)
	
	policy_ix = policy_ix.astype(int)
	policy = np.zeros(policy_ix.shape)
	ix = -1
	for row in PARAMS["income_states"]:
		ix += 1
		jx = -1
		for col in PARAMS["asset_states"]:
			jx += 1
			policy[ix, jx] = PARAMS["action_set"][policy_ix[ix, jx]]

	ergodic_distr = solve_distr(policy, params)
	#ipdb.set_trace()
	net_asset_demand = check_mc(params, policy, ergodic_distr.T)
	
	return(net_asset_demand)

root_rate = sp.optimize.bisect(solve_model, -0.1, 3, args = PARAMS)

root_rate = sp.optimize.newton(solve_model, 1, fprime=None, args=(PARAMS,))

rate_guess = root_rate
x = np.zeros((2,3,4))

x[0,0,:] = 4
# TODO: Give TauxschenHussey correct params
	# Implement Floden weights

x = np.array([[[1,2,3], [6,5,4]], [[8,10,13], [90,10,11]]])
np.argmax(x, 2)
np.max(x, 2)
consumption = calc_consumption(params, rate_guess)
# TODO numba


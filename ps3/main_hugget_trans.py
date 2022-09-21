# solve for a transition in the hugget economy

import numpy as np
import sys
sys.path.append('../')
from src.vfi_funcs import value_function_iterate
from src.reward_funcs import get_util, calc_consumption, find_asset_grid
from src.distribution_funcs import solve_distr, check_mc, policy_ix_to_policy
import ipdb
from numba import jit, njit, prange
import warnings
import time
import pandas as pd
import plotly.express as px
import scipy as sp

PARAMS = {
	"disc_fact": 0.993362,
	"risk_aver": 1.5,
	"policy_bc": np.linspace(-3, -6, 25),
	"T": 100, # TODO: check if sufficient
	"asset_net": 0,
	"asset_max": 10,
	"value_func_tol": 1e-6,
	"distr_tol": 1e-8,
	"mc_tol": 1e-3,
	"inc_grid_size": 2, 
	"action_grid_size": 1000, 
	"transition_matrix": np.array([[0.5, 0.5], [1 - 0.925, 0.925]]),
	"income_states": np.array([0.1, 1])
}

def solve_hh(rate, reward_matrix, asset_states, V_guess, params):
		
	arbitrary_slice_index = 0

	V, pol = value_function_iterate(V_guess[:,:, arbitrary_slice_index], PARAMS["transition_matrix"], reward_matrix, params["income_states"], asset_states, params["disc_fact"], params["value_func_tol"])
	# TODO: implement endoegnous grid method
	return(V, pol)

def solve_ss(rate_guess, borrow_constr, V_guess, params):
	''' Solves steady state

	borrow_constr : in steady state, borrow constraint should be a fixed scalar
	'''
	if rate_guess <= 0:
		return(-1) # Bad guess, don't continue
	if rate_guess >= 1/params["disc_fact"]-1:
		return(1) # Bad guess, don't continue
	
	asset_states = find_asset_grid(params, borrow_constr[0])
	action_set = np.copy(asset_states)

	reward_matrix = get_util(calc_consumption(params, rate_guess, borrow_constr), params)
	
	V, policy_ix = solve_hh(rate_guess, reward_matrix, asset_states, V_guess, params)
	policy = np.zeros(policy_ix.shape)
	
	for row in range(params["income_states"].shape[0]):
		for col in range(asset_states.shape[0]):
			policy[row, col] = action_set[policy_ix[row, col]]
	distr = solve_distr(policy_ix, params)
	net_asset_demand = check_mc(params, policy, distr.T)
	return(net_asset_demand, V, policy, distr)

def find_ss_rate(borrow_constr, V_guess, params):
	'''
	Find the interest rate that sets the steady state economy in equilibrium.
	'''
	x0 = 0.001
	x1 = 1/params["disc_fact"]-1-0.001
	try:
		root_rate= sp.optimize.newton(lambda x: solve_ss(x, borrow_constr, V_guess, params)[0], x0 = x0, x1 = x1 , tol = params["mc_tol"])
	except TypeError as e:
		print("Probably got: TypeError: 'int' object is not subscriptable. This means the grid is too coarse I think. INcreasing the size seems to help.")
		raise e
	
	return(root_rate)

##### test ground ###################
V_guess = np.full((
			PARAMS["inc_grid_size"],
			PARAMS["action_grid_size"], 
			PARAMS["action_grid_size"]), -100.0)
rate_pre = find_ss_rate(np.repeat(-3,2), V_guess, PARAMS)
rate_post = find_ss_rate(np.repeat(-6,2), V_guess, PARAMS)
#####################################

# TODO: next, check that the below works. 
# TODO: note that asset grid is no longer, necessarily, the same as the action grid. 

def	solve_transition(rate_guess, rate_post, V_post, params):
	''' Guess an interest rate transition vector, and output net asset demand

	Given an interest rate guess vector, the implied hh behavior and subsequent distribuiton and net asset demand is calculated along the whole transition. 
	'''
	action_axis = 2
	pol_ix = [] #np.empty(params["T"]-1)
	distr = []  #np.empty(pol_ix.shape)
	net_asset_demand = [] #np.empty(pol_ix.shape)
	borrow_constr = params["policy_bc"]
	income_states = params["income_states"]
	np.append(rate_guess, rate_post)
	
	V = V_post
	for t in reversed(range(1, params["T"]-1)):
		# Within loop, find the root finding rate for each t, starting backwards
		V_next = V
		try:
			r_guess = rate_guess[t]
		except IndexError as e:
			ipdb.set_trace()
			raise e
		
		bc_t = borrow_constr[min(len(borrow_constr)-1,t)]
		bc_t_next = borrow_constr[min(len(borrow_constr)-1,t+1)]
		bc = np.array([bc_t, bc_t_next])
		reward_matrix = get_util(calc_consumption(params, r_guess, bc), params)
		asset_states = find_asset_grid(params, bc_t)
		action_set = find_asset_grid(params, bc_t_next)
		asset_states_next = np.copy(action_set)
		
		V, pol = solve_hh(r_guess, reward_matrix, asset_states_next, np.expand_dims(V_next,2), params)
		
		pol_ix.append(pol)
		
		distr.append(solve_distr(pol_ix[-1], params))
		policy = policy_ix_to_policy(pol_ix[-1], income_states, asset_states, action_set)
		net_asset_demand.append(check_mc(params, policy.T, distr[-1])) # TODO: dislike that I need to transpose
		# TODO: SOlve for r vector one by one . HERE
	try:
		sum_of_squares = np.sum(np.asarray(net_asset_demand)**2)
	except Exception as e:
		ipdb.set_trace()
		raise 
	return(sum_of_squares, net_asset_demand, pol_ix, distr)

def find_transition_eq(params, rate_pre, rate_post, V_post):
	V_guess = np.full((
			params["inc_grid_size"],
			params["action_grid_size"], 
			params["action_grid_size"]), -100.0)
	if rate_pre is None:
		rate_pre = find_ss_rate(np.repeat(params["policy_bc"][0],2), V_guess, params)
	if rate_post is None:
		rate_post = find_ss_rate(np.repeat(params["policy_bc"][-1],2), V_guess, params)
	if V_post is None:
		V_post = solve_ss(rate_post, np.repeat(params["policy_bc"][-1],2), V_guess, params)[1]
	
	r_guess = np.linspace(rate_pre, rate_post, params["T"]-1) # Last value is appended with rate_post inside objective function.
	rate_path = sp.optimize.minimize(lambda x: solve_transition(x, rate_post, V_post, params)[0], x0 = r_guess, tol = params["mc_tol"])

	rate_path = np.append(rate_pre, rate_path)
	rate_path = np.append(rate_path, rate_post)

	return(rate_path)

rate_path = find_transition_eq(PARAMS, rate_pre, rate_post, None)
net_asset_demand, pol, distr = solve_transition(rate_path, rate_post, PARAMS):


	a = np.array([[1,2],[3, 4]])
	a[None, 0]
	

# plot the interest rate path

# reproduce the distribution over welfate gains in consumption equivalent units

# Endogenous grid method


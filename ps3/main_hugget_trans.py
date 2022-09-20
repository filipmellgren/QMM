# solve for a transition in the hugget economy

import numpy as np
import sys
sys.path.append('../')
from src.vfi_funcs import value_function_iterate
from src.reward_funcs import get_util, calc_consumption, find_asset_grid
from src.distribution_funcs import solve_distr, check_mc
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

def solve_hh(rate, reward_matrix, asset_states, params):

	V = np.full((
	params["inc_grid_size"],
	asset_states.shape[0], 
	asset_states.shape[0]), -100.0)
		
	arbitrary_slice_index = 0

	V, pol = value_function_iterate(V[:,:, arbitrary_slice_index], PARAMS["transition_matrix"], reward_matrix, params["income_states"], asset_states, params["disc_fact"], params["value_func_tol"])
	# TODO: implement endoegnous grid method
	return(V, pol)

def solve_ss(rate_guess, borrow_constr, params):

	if rate_guess <= 0:
		return(-1) # Bad guess, don't continue
	if rate_guess >= 1/params["disc_fact"]-1:
		return(1) # Bad guess, don't continue

	asset_states = find_asset_grid(params, borrow_constr)
	action_set = np.copy(asset_states)

	reward_matrix = get_util(calc_consumption(params, rate_guess, borrow_constr), params)
	V, policy_ix = solve_hh(rate_guess, reward_matrix, asset_states, params)
	policy = np.zeros(policy_ix.shape)
	
	for row in range(params["income_states"].shape[0]):
		for col in range(asset_states.shape[0]):
			policy[row, col] = action_set[policy_ix[row, col]]
	distr = solve_distr(policy, policy_ix, params)
	net_asset_demand = check_mc(params, policy, distr.T)
	return(net_asset_demand, V, policy, distr)

def find_ss_rate(borrow_constr, params):
	'''
	Find the interest rate that sets the steady state economy in equilibrium.
	'''
	x0 = 0.001
	x1 = 1/params["disc_fact"]-1-0.001
	root_rate= sp.optimize.newton(lambda x: solve_ss(x, borrow_constr, params)[0], x0 = x0, x1 = x1 , tol = params["mc_tol"])
	return(root_rate)

##### test ground ###################
rate_pre = find_ss_rate(-3, PARAMS)
rate_post = find_ss_rate(-6, PARAMS)
#####################################


def	solve_transition(rate_guess, params):
	'''

	'''
	pol = np.empty(params["T"]-2)
	distr = np.empty(pol.shape)
	net_asset_demand = np.empty(pol.shape)
	for t in reversed(range(1, params["T"]-1)):
		pol[t] = solve_hh(r_guess[t])
		distr[t] = solve_distr(pol[t])
		net_asset_demand[t] = check_mc(params, pol[t], distr[t])
		if abs(net_asset_demand[t]) > params["mc_tol"]:
			return(abs(net_asset_demand[t]))

	max_net_demand = np.max(net_asset_demand)

	return(max_net_demand, net_asset_demand, pol, distr)

def find_transition_eq():
	ss_pre = find_ss_rate(params["policy_bc"][0], params)
	ss_post = find_ss_rate(params["policy_bc"][-1], params)
	r_guess = np.linspace(ss_pre["rate"], ss_post["rate"], params["T"])
	transition = sp.optimize(lambda x: solve_transition(x, params)[0], x0 = r_guess, tol = params["mc_tol"])
	return(transition)
	

# plot the interest rate path

# reproduce the distribution over welfate gains in consumption equivalent units

# Endogenous grid method


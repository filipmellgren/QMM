# solve for a transition in the Hugget economy

# TODO: something goes wrong after 25 time periods. Possibly earlier too. 
# TODO: not sure I need to interpolate? I already have the mapping in a way.
# Maybe error is how I use the policy later.
# SHould run functions one by one to see what happens in each 

import numpy as np
import sys
sys.path.append('../')
from src.vfi_funcs import value_function_iterate, egm
from src.reward_funcs import get_util, calc_consumption, find_asset_grid
from src.distribution_funcs import solve_distr, check_mc, policy_ix_to_policy
import ipdb
from numba import jit, njit, prange
import warnings
import time
import pandas as pd
import plotly.express as px
import scipy as sp
import pandas as pd
from interpolation import interp
from scipy import interpolate


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

PARAMS["V_guess"] = np.full((
			PARAMS["inc_grid_size"],
			PARAMS["action_grid_size"], 
			PARAMS["action_grid_size"]), -100.0)

def solve_hh(rate, borrow_constr, reward_matrix, consumption_matrix, asset_states, action_set, V_guess, params, method):
		
	arbitrary_slice_index = 0

	if method == "VFI":
		# Returns index of policy. Make consistent
		V, pol_ix = value_function_iterate(V_guess[:,:, arbitrary_slice_index], PARAMS["transition_matrix"], reward_matrix, params["income_states"], asset_states, params["disc_fact"], params["value_func_tol"])
		pol = policy_ix_to_policy(pol_ix, params["income_states"], asset_states, action_set)
		pol = pol.T
	
	if method == "EGM":	
		# Return actual policy, not index
		
		action_states = find_asset_grid(params, borrow_constr[1])
		V_guess = V_guess[:,:, arbitrary_slice_index].T
		V, pol = egm(V_guess, PARAMS["transition_matrix"], action_states, asset_states, rate, params, tol = 1e-6)
		# Transform policy back to the exogenous grid for compatibility with distribution iteration
		for s in range(pol.shape[1]):
			pol_s = pol[:,s]
			pol[:,s] = action_states[abs(pol_s[None, :] - action_states[:, None]).argmin(axis=0)]
	return(V, pol)

def solve_ss(rate_guess, borrow_constr, V_guess, params, method = "VFI"):
	''' Solves steady state

	borrow_constr : in steady state, borrow constraint should be a fixed scalar
	'''
	if rate_guess <= 0:
		return([-100]) # Bad guess, don't continue
	if rate_guess >= 1/params["disc_fact"]-1:
		return([100]) # Bad guess, don't continue
	
	asset_states = find_asset_grid(params, borrow_constr[0])
	action_set = np.copy(asset_states)
	
	consumption_matrix = calc_consumption(params, rate_guess, borrow_constr)
	reward_matrix = get_util(consumption_matrix, params)
	
	V, policy = solve_hh(rate_guess, borrow_constr, reward_matrix, consumption_matrix, asset_states, action_set, V_guess, params, method)
	assert not np.any(np.isnan(policy)), "There are nan values in the policy matrix"

	distr = solve_distr(policy, action_set, params)
	net_asset_demand = check_mc(params, policy, distr)
	return(net_asset_demand, V, policy, distr)

def find_ss_rate(borrow_constr, V_guess, params, method = "VFI"):
	'''
	Find the interest rate that sets the steady state economy in equilibrium.
	'''
	x0 = 0.001
	x1 = 1/params["disc_fact"]-1-0.001
	try:
		root_rate= sp.optimize.newton(lambda x: solve_ss(x, borrow_constr, V_guess, params, method)[0], x0 = x0, x1 = x1 , tol = params["mc_tol"])
	except TypeError as te:
		print("Probably got: TypeError: 'int' object is not subscriptable. This means the grid is too coarse I think. Increasing the size seems to help.")
		raise te
	except RuntimeError:
		root_rate = sp.optimize.bisect(lambda x: solve_ss(x, borrow_constr, V_guess, params, method), a = x0, b = x0, xtol = params["mc_tol"]) 
	
	return(root_rate)

bc_ss = np.repeat(PARAMS["policy_bc"][0],2)
tmp = solve_ss(0.001, bc_ss, PARAMS["V_guess"], PARAMS, "EGM")

rate_pre = find_ss_rate(np.repeat(PARAMS["policy_bc"][0],2), PARAMS["V_guess"], PARAMS, method = "EGM")

rate_pre = find_ss_rate(np.repeat(PARAMS["policy_bc"][0],2), PARAMS["V_guess"], PARAMS, method = "VFI")

import matplotlib.pyplot as plt
plt.plot(endog_asset, exog_action, color ="red")
plt.show()

def	solve_transition(rate_guess, t, V_post, params):
	''' Guess an interest rate transition vector, and output net asset demand

	Given an interest rate guess vector, the implied hh behavior and subsequent distribuiton and net asset demand is calculated along the whole transition. 
	'''
	V_next = V_post
	borrow_constr = params["policy_bc"]
	income_states = params["income_states"]
	pol_list = []
	distr = []
	
	bc_t = borrow_constr[min(len(borrow_constr)-1,t)] # TODO: something wacky happens when borrow _constr - 1 is min
	bc_t_next = borrow_constr[min(len(borrow_constr)-1,t+1)]
	bc = np.array([bc_t, bc_t_next])
	consumption_matrix = calc_consumption(params, rate_guess, bc)
	reward_matrix = get_util(consumption_matrix, params)
	asset_states = find_asset_grid(params, bc_t)
	action_set = find_asset_grid(params, bc_t_next)
	asset_states_next = np.copy(action_set)
	
	V, pol = solve_hh(rate_guess, bc_t, reward_matrix, consumption_matrix, asset_states_next, action_set, np.expand_dims(V_next,2), params, method = "VFI")
	
	pol_list.append(pol)
	distr.append(solve_distr(pol_list[-1], action_set, params)) 
	net_asset_demand = check_mc(params, policy, distr[-1]) 
	return(net_asset_demand)

def find_transition_eq(params, rate_pre = None, rate_post = None, V_post = None):
	''' Find path of the interest rate between the two steady states.

	Can take as input pre and post ss interest rates. Default is to compute them.
	'''
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
	
	rate_next = rate_post
	rate_path = []
	for t in reversed(range(1, params["T"]-1)):
		# Within loop, find the root finding rate for each t, starting backwards
		root_rate_t = sp.optimize.newton(lambda x: solve_transition(x, t, V_post, params), x0 = rate_next, x1 = rate_next - 0.005, tol = params["mc_tol"])
		rate_path.append(root_rate_t)
		rate_next = root_rate_t
		print("Found this rate: " + str(root_rate_t) + " at time: " + str(t))

	rate_path = np.append(rate_path, rate_pre)
	rate_path = np.append(rate_post, rate_path)

	return(rate_path)

rate_path = find_transition_eq(PARAMS)

# ANALYSIS ####################################################

df = pd.DataFrame(rate_path, columns = ["rate"])
df["time"] = abs(df.index - PARAMS["T"])-1

fig = px.line(df, x="time", y="rate", 
	title='Interest rate path <br><sup> ' + "Value function iteration" + ' </sup>',
	template = 'plotly_white')
fig.add_vline(x=len(PARAMS["policy_bc"])-1, line_width=2, line_dash="dash")
fig.show()
fig.write_image('figures/rate_path.png')

# reproduce the distribution over welfare gains in consumption equivalent units
def calc_welfare():
	alpha = 0 # A solution to an equality it seems. Can sometimes be backed out. 
	consumption_units_gained = alpha
	return(consumption_units_gained)

# Endogenous grid method



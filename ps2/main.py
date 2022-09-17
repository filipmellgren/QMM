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
from numba import jit, njit, prange
from numba.typed import Dict
import warnings
import time
import pandas as pd
import plotly.express as px

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
	"action_grid_size": 1000, 
}

def floden_w(ar1):
	'''
	Implements Flodén weighting
	'''
	weight = 0.5 + ar1/4
	return(weight)

floden_sd = floden_w(PARAMS["ar1"]) * np.sqrt(PARAMS["shock_var"]) + (1- floden_w(PARAMS["ar1"]))* np.sqrt(PARAMS["shock_var"]/(1-PARAMS["ar1"]**2))

income_states, transition_matrix = tauchenhussey(
	PARAMS["inc_grid_size"], # number of nodes for Z
	PARAMS["shock_mean"], # unconditional mean of process
	PARAMS["ar1"], # rho
	np.sqrt(PARAMS["shock_var"]), # std. dev. of epsilons
	floden_sd) # std. dev. used to calculate Gaussian 
PARAMS["income_states"] = np.copy(income_states[0])
PARAMS["asset_states"] = np.logspace(start = 0, stop = np.log10(PARAMS['asset_max'] - PARAMS['borrow_c']+1), num = PARAMS['action_grid_size']) + PARAMS['borrow_c'] -1
PARAMS["action_set"] = np.copy(PARAMS["asset_states"])
PARAMS["transition_matrix"] = transition_matrix

def get_util(consumption, params):
	'''
	Calculate agent dirtect utility. 
	TODO: implement option to specify functional form
	'''
	
	utility = (consumption**(1-params["risk_aver"])-1)/(1-params["risk_aver"])
	utility[consumption<0] = -np.inf 

	return(utility)

def calc_consumption(params, rate):
	income_states = params["income_states"]
	asset_states = params["asset_states"]
	action_set = params["action_set"]
	income, asset, saving = np.meshgrid(income_states, asset_states, action_set, sparse = True, indexing='ij')
	consumption = income + asset * (1+rate) - saving
	return(consumption)

def solve_hh(interest_rate, reward_matrix, params):
	# HH maximize. Considers for all states what the optimal action is and updates their policies based on this.
	# TODO: HH can be a class with V, policy as attributes
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
	V = value_function_iterate(V[:,:, arbitrary_slice_index], transition_matrix, reward_matrix, params["income_states"], params["asset_states"], params["action_set"], params["disc_fact"], params["value_func_tol"])
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
			# Transition vector is the transition probability to a state (col)  from all possible states in the rows
			transition_vector = params["transition_matrix"][:,it.multi_index[income_dim]] # 3 by 1
	
			a_k = params["action_set"][it.multi_index[action_dim]] # 1 by 1
			
			indicator = a_k == policy[it.multi_index[income_dim],:] # 1000 by 1
	
			distr_upd[it.multi_index] = indicator.T @ distr_guess @ transition_vector # 1 by 1
		converged = np.max(distr_upd - distr_guess) < params["distr_tol"]
		distr_guess = np.copy(distr_upd)
	
	sum_distribution = np.sum(distr_guess)
	assert abs(sum_distribution - 1)<0.0001, "Distribution sums to " + str(sum_distribution)
	assert np.all(distr_guess >= 0), "Negative values in distribution"

	return(distr_upd)

@jit(nopython=True, fastmath=True, parallel = True)
def value_function_iterate(V, transition_matrix, reward_matrix, income_states, asset_states, actions,  disc_factor, tol):
	# V here is just the states
	V_new = np.copy(V)
	POL = np.zeros(V_new.shape, dtype=np.int16)
	diff = 100.0
	
	while diff > tol:
		diff = 0.0

		for inc_ix in range(income_states.shape[0]):
			P = transition_matrix[inc_ix, :]
			V = np.copy(V_new)
			exp_val = np.dot(P, V)
			for ass_ix in prange(asset_states.shape[0]):
				v = V[inc_ix, ass_ix]
				V_new_cands = reward_matrix[inc_ix, ass_ix,:] + disc_factor * exp_val
				pol = np.argmax(V_new_cands)
				POL[inc_ix, ass_ix] = pol
				v_new = V_new_cands[pol]
				V_new[inc_ix, ass_ix] = v_new
				diff = max(diff, abs(v - v_new))

	return(V, POL)

def find_expval(exp_val, V, POL, actions, income_states, transition_matrix, inc_ix):
	for act_ix in prange(actions.shape[0]): #actions: # Find expected value of each action
		for inc_ix2 in range(income_states.shape[0]):#income_states:
			exp_val[act_ix] += transition_matrix[inc_ix, inc_ix2] * V[inc_ix2, act_ix, POL[inc_ix2, act_ix]]
	return(exp_val)

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

def solve_model(rate_guess, params):
	reward_matrix = get_util(calc_consumption(params, rate_guess), params)
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
	net_asset_demand = check_mc(params, policy, ergodic_distr.T)
	
	return(net_asset_demand, V, policy, ergodic_distr)

if __name__ == "__main__":
	#root_rate = sp.optimize.bisect(solve_model, -1, 1/PARAMS["disc_fact"]-1-0.15, xtol = PARAMS["mc_tol"], args = PARAMS) # Bisection method option
	root_rate= sp.optimize.newton(lambda x: solve_model(x, PARAMS)[0], x0 = -0.5, x1 =  1/PARAMS["disc_fact"]-1-0.15, tol = PARAMS["mc_tol"])
	net_assets, V, policy, distr = solve_model(root_rate, PARAMS)

	with open('root_rate.txt', 'w') as f:
	  f.write('%d' % root_rate)

	income_states_df = pd.DataFrame(income_states,columns = ["low","medium", "high"])
	income_states_df.to_csv("income_states.csv")

	df = pd.DataFrame(policy.T,columns = ["low", "medium", "high"])
	df["assets"] = PARAMS["action_set"][df.index]
	df = pd.melt(df, id_vars=["assets"], value_vars=["low", "medium", "high"], var_name='income_type', value_name='policy')

	df_income = {'income_type': ["low", "medium", "high"], 'income_value': income_states.tolist()[0]}
	df_income = pd.DataFrame(data=df_income)
	df = df.merge(df_income, left_on='income_type', right_on='income_type')
	df["consumption"] = df["income_value"] + df["assets"] * (1 + root_rate) - df["policy"]
	df.to_csv("value_function.csv")


	fig = px.line(df, x="assets", y="consumption", color = "income_type", title='Consumption policy', template = 'plotly_white', color_discrete_sequence=["#2FF3E0", "#F8D210", "#FA26A0"])

	fig.write_image('figures/consumption_by_income.png')

	# Analyze performance of code. For fun. 
	PARAMS["action_grid_size"] = 0
	N_tests = 10
	perf = np.zeros((N_tests, 3))

	for i in range(N_tests):
		PARAMS["action_grid_size"] += 100
		PARAMS["asset_states"] = np.logspace(start = 0, stop = np.log10(PARAMS['asset_max'] - PARAMS['borrow_c']+1), num = PARAMS['action_grid_size']) + PARAMS['borrow_c'] -1
		PARAMS["action_set"] = np.copy(PARAMS["asset_states"])
		st = time.time()
		root_rate= sp.optimize.newton(lambda x: solve_model(x, PARAMS)[0], x0 = -0.5, x1 =  1/PARAMS["disc_fact"]-1-0.15, tol = PARAMS["mc_tol"])
		et = time.time()
		perf[i, 0] = PARAMS["action_grid_size"]
		perf[i, 1] = root_rate
		perf[i, 2] = et - st

	df = pd.DataFrame(perf,columns = ["size","solution", "time"])

	fig = px.scatter(df, x="size", y="time", color = "solution", 
		title='Time to find root_rate', template = 'plotly_white',
		labels=dict(size = "Asset grid points", time="Seconds", solution = "root_rate"))

	fig.write_image('figures/time_to_converge.png')

	# PART 2
	net_assets, V, policy, distr = solve_model(0, PARAMS)
	# Net assets is worth -3 goods. Since there are 10 money, 
	distr.shape


	# For fun, changing the variance of the income risk:

	PARAMS["shock_var"] = 0.02

	floden_sd = floden_w(PARAMS["ar1"]) * np.sqrt(PARAMS["shock_var"]) + (1- floden_w(PARAMS["ar1"]))* np.sqrt(PARAMS["shock_var"]/(1-PARAMS["ar1"]**2))

	income_states, transition_matrix = tauchenhussey(
		PARAMS["inc_grid_size"], # number of nodes for Z
		PARAMS["shock_mean"], # unconditional mean of process
		PARAMS["ar1"], # rho
		np.sqrt(PARAMS["shock_var"]), # std. dev. of epsilons
		floden_sd) # std. dev. used to calculate Gaussian 
	PARAMS["income_states"] = np.copy(income_states[0])
	PARAMS["asset_states"] = np.logspace(start = 0, stop = np.log10(PARAMS['asset_max'] - PARAMS['borrow_c']+1), num = PARAMS['action_grid_size']) + PARAMS['borrow_c'] -1
	PARAMS["action_set"] = np.copy(PARAMS["asset_states"])
	PARAMS["transition_matrix"] = transition_matrix

	root_rate= sp.optimize.newton(lambda x: solve_model(x, PARAMS)[0], x0 = -0.5, x1 =  1/PARAMS["disc_fact"]-1-0.15, tol = PARAMS["mc_tol"])




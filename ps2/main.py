# Problem set 2, money in the Huggett model
# Author: filip.mellgren@su.se

# LONG TERM ####
# TODO: implement the endogenous grid method
# TODO: check accuracy of the solution
# TODO: accelerator methods

import numpy as np
import scipy.stats as st
import scipy as sp
import sys
sys.path.append('../')
from src.tauchenhussey import tauchenhussey
from src.vfi_funcs import value_function_iterate
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
PARAMS["income_states"] = np.exp(income_states[0])
PARAMS["asset_states"] = np.logspace(start = 0, stop = np.log10(PARAMS['asset_max'] - PARAMS['borrow_c']+1), num = PARAMS['action_grid_size']) + PARAMS['borrow_c'] -1
PARAMS["action_set"] = np.copy(PARAMS["asset_states"])
PARAMS["transition_matrix"] = transition_matrix

def get_util(consumption, params):
	'''
	Calculate agent dirtect utility. 
	TODO: implement option to specify functional form
	'''
	utility = np.zeros(consumption.shape)
	with np.errstate(invalid='ignore'):
		utility = (np.power(consumption, 1-params["risk_aver"])-1)/(1-params["risk_aver"])
	utility[consumption<0] = -np.inf 

	assert utility[consumption.shape[0]-1, consumption.shape[1]-1, 0] == np.max(utility), "Expected max utility is not max."
	return(utility)

def calc_consumption(params, rate):
	income_states = params["income_states"]
	asset_states = params["asset_states"]
	action_set = params["action_set"]
	
	try:
		income, asset, saving = np.meshgrid(income_states, asset_states, action_set, sparse = True, indexing='ij')
	except SystemError:
		print("SystemError, what the @*!$€ is wrong?")
		ipdb.set_trace()
	
	consumption = income + asset * (1+rate) - saving

	# Check, all else equal, that consumption increases with higher values
	arbitrary_index_ass = np.random.randint(0, high=asset_states.shape[0]-1)
	arbitrary_index_act = np.random.randint(0, high=action_set.shape[0]-1)

	income_dim = consumption[:,arbitrary_index_ass,arbitrary_index_act]
	asset_dim = consumption[0,:,arbitrary_index_act]
	savings_dim = consumption[0,arbitrary_index_ass,:]

	if not np.all(income_dim[1:] > income_dim[:-1]):
		ipdb.set_trace()
		pass
	assert np.all(income_dim[1:] > income_dim[:-1]), "Consumption not increasing in income"
	
	if not np.all(asset_dim[1:] > asset_dim[:-1]):
		ipdb.set_trace()
		pass
	assert np.all(asset_dim[1:] > asset_dim[:-1]), "Consumption not increasing in assets"
	assert np.all(savings_dim[1:] <= savings_dim[:-1]), "Consumption not decreasing with savings"
	assert consumption.shape == (income_states.shape[0], asset_states.shape[0], action_set.shape[0])
	assert consumption[0,0,action_set.shape[0]-1] == np.min(consumption), "Expected min consumption is not actual min consumption"
	return(consumption)

def solve_hh(interest_rate, reward_matrix, params):
	# HH maximize. Considers for all states what the optimal action is and updates their policies based on this.
	# TODO: HH can be a class with V, policy as attributes
	# Matrix dimension/depth is income, rows are asset dimension, columns are action dimension
	
	V = np.full((
		params["inc_grid_size"],
		params["action_grid_size"], 
		params["action_grid_size"]), -100.0)
		
	arbitrary_slice_index = 0
	V, pol = value_function_iterate(V[:,:, arbitrary_slice_index], transition_matrix, reward_matrix, params["income_states"], params["asset_states"], params["disc_fact"], params["value_func_tol"])
	return(V, pol)

@jit(nopython=True, parallel = True)
def iterate_distr(distr_guess, policy_ix, transition_matrix, inc_size, asset_size, tol):
	diff = 100.0
	while diff > tol:
		diff = 0.0
			# Transition vector is the transition probability to a state (col)  from all possible states in the rows
		for sl in prange(inc_size):
			transition_vector = transition_matrix[:,sl] # From sl to sj # TODO: correct index?
			for ak in range(asset_size):
				#ak_val = params["action_set"][ak]
				indic = ak == policy_ix
				d_new = np.sum(np.dot((indic * distr_guess), transition_vector))
				d_prev = distr_guess[ak, sl]
				diff = max(diff, abs(d_new - d_prev))
				distr_guess[ak, sl] = d_new
		distr_guess = distr_guess/np.sum(distr_guess) # TODO: this is super hacky! It is needed bc I think the distribution starts to drift away from 1. 
	return(distr_guess)

def solve_distr(policy, policy_ix, params):
	'''
	Given interest rate, houesehold policies, and exogenous income process, what distribution do we end up with?
	'''
	policy = policy.T
	policy_ix = policy_ix.T
	policy_ix = policy_ix.astype(int) # Want to be sure of this when using in Numba
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

	assert policy_ix.shape == (params["action_grid_size"], params["inc_grid_size"]), "Policy index not of expected dimension. It is: " + str(policy_ix.shape)
	'''
	while diff > tol:
		diff = 0.0
		i += 1
			# Transition vector is the transition probability to a state (col)  from all possible states in the rows
		for sl in range(params["inc_grid_size"]):
			transition_vector = transition_matrix[:,sl] # From sl to sj # TODO: correct index?
			for ak in range(params["action_grid_size"]):
				#ak_val = params["action_set"][ak]
				indic = np.isclose(ak, policy_ix)
				d_new = np.sum(np.dot((indic * distr_guess), transition_vector))
				d_prev = distr_guess[ak, sl]
				diff = max(diff, abs(d_new - d_prev))
				distr_guess[ak, sl] = d_new
		distr_guess = distr_guess/np.sum(distr_guess) # TODO: this is super hacky! It is needed bc I think the distribution starts to drift away from 1. 
	'''
	distr_guess = iterate_distr(distr_guess, policy_ix, transition_matrix, params["inc_grid_size"], params["action_grid_size"], tol)

	sum_distribution = np.sum(distr_guess)

	assert abs(sum_distribution - 1)<0.001, "Distribution sums to " + str(sum_distribution)
	assert np.all(distr_guess >= 0), "Negative values in distribution"
	return(distr_guess)
	'''
	
				for sj in range(params["inc_grid_size"]):
					p = transition_matrix[sj, sl]
					for ai in range(params["action_grid_size"]):
						d_guess = distr_guess[ai, sj]
						indic = np.isclose(ak_val, policy[ai, sj])
						d_new += d_guess * p * indic
				d_prev = distr_guess[ak, sl]
				diff = max(diff, abs(d_new - d_prev))
				distr_guess[ak, sl] = d_new
	sum_distribution = np.sum(distr_guess)
	assert abs(sum_distribution - 1)<0.001, "Distribution sums to " + str(sum_distribution)
	assert np.all(distr_guess >= 0), "Negative values in distribution"
	print(i)
	return(distr_guess)
	
		for v in it:
			transition_vector = transition_matrix[:,it.multi_index[income_dim]] # 3 by 1
	
			a_k = actions[it.multi_index[action_dim]] # 1 by 1
			
			indicator = a_k == policy[it.multi_index[income_dim],:] # 1000 by 1
	
			distr_upd[it.multi_index] = indicator.T @ distr_guess @ transition_vector # 1 by 1
			d_new = distr_upd[it.multi_index]
			d_prev = distr_guess[it.multi_index]
			diff = max(diff, abs(d_new - d_prev))
		distr_guess = np.copy(distr_upd)
	print(i)
	sum_distribution = np.sum(distr_guess)
	assert abs(sum_distribution - 1)<0.0001, "Distribution sums to " + str(sum_distribution)
	assert np.all(distr_guess >= 0), "Negative values in distribution"
	return(distr_guess)
	'''

def construct_markov_chain(transition_matrix, policy_ix, params):
	'''
	This function is complicated.
	It repeats the asset states once per income state by the rows and cols to create an "(N x S) x (N x S)" matrix with values being I * p where I is an indicator whether the row a and s maps onto the col a, and p is the probability of going from row to col s. 
	It is designed such that the income state 0 comes first with all asset states, the state 0 is then followed by state 1 and all asset states once more, and so on, for both the rows and the cols.
	'''
	
	inc_s_num = params["income_states"].shape[0] # "S"
	a_s_num = params["asset_states"].shape[0] # "N"

	pol, act = np.meshgrid(policy_ix, range(a_s_num), sparse = True, indexing='ij') # First income 0, all asset states, then income 1, all assets, etc. 

	indicator_matrix = pol == act
	indicator_matrix = np.tile(indicator_matrix, inc_s_num) # repeat the Ind-matrix once for each income state
	repelem = lambda a, x, y: np.repeat(np.repeat(a, x, axis=0), y, axis=1)
	rep_probs = repelem(transition_matrix, a_s_num, a_s_num) # create block matrix of probabilities
	markov_chain = indicator_matrix * rep_probs

	row_sums = np.sum(markov_chain, axis = 1)
	assert np.all(np.isclose(row_sums, 1)), "Transition matrix is not right stochastic, given row, columns should sum to one."
	return(markov_chain)
def solve_distr_by_eigenvector(policy, policy_ix, params):
	# TODO: not complete and not sure it is workin 100%
	markov_chain = construct_markov_chain(transition_matrix, policy_ix, params)
	evals, evecs = np.linalg.eig(markov_chain)
	evec1 = evecs[:,np.isclose(evals, 1)]
	evec1 = abs(evec1 / evec1.sum())
	distr = evec1.reshape(params["inc_grid_size"], params["action_grid_size"]) # markov_chain gives first income 0 [all assets], then income 1 [all assets]
	#distr = evec1.reshape(params["action_grid_size"], params["inc_grid_size"])
	distr = distr.T # We want it to be assets x incomes (note we still need to be careful about reshaping to inc size x assets size)

	distr = distr.real
	# CHECK is real, sum to one, all nonnegative
	#distr = distr/sum(distr)
	return(distr)

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
	print(rate_guess)
	reward_matrix = get_util(calc_consumption(params, rate_guess), params)
	V, policy_ix = solve_hh(rate_guess, reward_matrix, params) # Given rate, what would HH do? (all households behave same way, but end up in different states)
	
	policy = np.zeros(policy_ix.shape)
	
	for row in range(PARAMS["income_states"].shape[0]):
		for col in range(PARAMS["asset_states"].shape[0]):
			policy[row, col] = PARAMS["action_set"][policy_ix[row, col]]

	ergodic_distr = solve_distr(policy, policy_ix, params)
	
	net_asset_demand = check_mc(params, policy, ergodic_distr.T)
	
	return(net_asset_demand, V, policy, ergodic_distr)

if __name__ == "__main__":
	try:
		root_rate= sp.optimize.newton(lambda x: solve_model(x, PARAMS)[0], x0 = 0, x1 =  1/PARAMS["disc_fact"]-1-0.003, tol = PARAMS["mc_tol"])
	except RuntimeError:
		print("Newton failed! Trying with bisection")
		root_rate = sp.optimize.bisect(lambda x: solve_model(x, PARAMS)[0], -1+0.1, 1/PARAMS["disc_fact"]-1, xtol = PARAMS["mc_tol"]) # Bisection method option

	net_demand, V, policy, distr = solve_model(root_rate, PARAMS)

	f = open('root_rate.txt', 'w')
	f.write(str(root_rate))
	f.close()

	f = open('income_states.txt', 'w')
	f.write(str(income_states[0]))
	f.close()

	fig = px.imshow(policy, aspect="auto", 
		labels=dict(x="Current assets", y="Current income", color="Savings"),
                x=PARAMS["asset_states"],
                y=['Low', 'Medium', 'High'],
                title = "Policy matrix <br><sup>Yes, it is stupid that incomes are shown the way they are</sup>")
	fig.write_image('figures/policy_matrix.png')

	fig = px.imshow(distr.T, aspect="auto", 
		labels=dict(x="Current assets", y="Current income", color="Density"),
		x=PARAMS["asset_states"],
		y=['Low', 'Medium', 'High'],
		title = "Ergodic distribution matrix <br><sup>Yes, it is stupid that incomes are shown the way they are</sup>")
	fig.write_image('figures/ergodic_distr.png')

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
		root_rate= sp.optimize.newton(lambda x: solve_model(x, PARAMS)[0], x0 = 0, x1 =  1/PARAMS["disc_fact"]-1-0.003, tol = PARAMS["mc_tol"])
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

	new_root_rate= sp.optimize.newton(lambda x: solve_model(x, PARAMS)[0], x0 = 0, x1 =  1/PARAMS["disc_fact"]-1, tol = PARAMS["mc_tol"])

	f = open('new_root_rate.txt', 'w')
	f.write(str(new_root_rate))
	f.close()



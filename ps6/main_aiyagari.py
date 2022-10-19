# Ayiagari model
import numpy as np
import scipy as sp
import sys
sys.path.append('../')
from src.tauchenhussey import tauchenhussey, floden_basesigma 
from src.vfi_funcs import egm
from src.distribution_funcs import kieran_finder
import pandas as pd
import ipdb
import plotly.express as px
from numba import jit, njit, prange
from scipy.interpolate import griddata
import time

import plotly.io as pio
pio.templates.default = "plotly_white"

def market_clearing(r_in):

	min_assets = -1
	max_assets = 150
	asset_grid = np.logspace(
		start = 0, 
		stop = np.log10(max_assets+2), 
		num = 1500) -2

	alpha = 0.3
	sigma = np.sqrt(0.04)
	rho = 0.94
	mu = 0
	inc_size = 5
	baseSigma = floden_basesigma(sigma, rho)
	log_income_states, transition_matrix = tauchenhussey(inc_size, mu,rho,sigma, baseSigma)
	income_states = np.exp(log_income_states[0,:])
	steady_distribution = np.linalg.matrix_power(transition_matrix, 1000)
	mean_income =  income_states @ steady_distribution[0,:]
	income_states = income_states / mean_income # Checked with Tom

	delta = 0.1
	K_firm = (alpha/(r_in + delta))**(1/(1-alpha))
	wage = (1 - alpha) * K_firm ** alpha
	income_states = wage * income_states
	
	params = {
	"disc_fact": 0.96,
	"risk_aver": 1.5,
	"income_states": income_states
	}

	policy = egm(transition_matrix, asset_grid, r_in, params, tol = 1e-6)
	policy_exog_grid = griddata(asset_grid, asset_grid, policy, method='nearest')

	ergodic_distribution  = EigenLambda(transition_matrix, asset_grid, income_states, policy_exog_grid)
	
	K_hh = np.sum(ergodic_distribution * policy_exog_grid)
	
	
	diff = K_hh-K_firm
	return(diff, ergodic_distribution, K_hh, policy_exog_grid, asset_grid, income_states)

def state_transition_matrix(transition_matrix, asset_grid, income_states, policy):
	""" Create Markov transition matrix

	Take the income transition matrix, tile it to be of size (NxS)(NxS) to have correct income transitions for each income state.

	Multiply by indicator of asset choices, from asset x income to asset, i.e. each row maps to a unique asset.
	# Checked with Tom, and identical when the policy is the asset grid repeated

	"""

	P = transition_matrix

	#PP = np.tile(P, (asset_grid.shape[0],asset_grid.shape[0])) # repeat the block matrix to shape 1500*1500
	#pol_flat = policy.flatten()
	#pol, ass = np.meshgrid(pol_flat, asset_grid, indexing='ij', sparse = True)
	#i_mat = (pol == ass) # From each income x asset to asset
	#i_mat = np.repeat(i_mat, income_states.shape[0], axis = 1)
	
	PP = np.repeat(np.repeat(P, asset_grid.shape[0], axis = 0), asset_grid.shape[0], axis = 1)
	pol_flat = policy.flatten("f")
	pol, ass = np.meshgrid(pol_flat, asset_grid, indexing='ij', sparse = True)
	i_mat = (pol == ass) # From each income x asset to asset
	assert np.all(np.isclose(np.sum(i_mat, axis = 1), 1))
	i_mat = np.tile(i_mat, income_states.shape[0])
	assert np.all(np.isclose(np.sum(PP * i_mat, axis = 1), 1))
	# Note, with row stochastic matrix minus identity, 
	# One solution will always be a constant vector. Not what we want
	# Therefore transpose the result
	PP = (PP * i_mat).T
	return(PP)

def EigenLambda(transition_matrix, asset_grid, income_states, policy):
	# TODO: should give an eigenvector given that we know the eigenvalue.
	# Tom's code does fortran like oprdering of the reshape
	
	#eigen_vec = np.linalg.solve(PP - np.identity(PP.shape[0]), np.zeros(PP.shape[0]))
	PP = state_transition_matrix(transition_matrix, asset_grid, income_states, policy)
	value, vector = sp.sparse.linalg.eigs(PP, k=1, sigma=1) # Decently fast. Search for one eigenvalue close to sigma.
	
	assert np.isclose(value, 1)
	vector = vector / np.sum(vector)
	vector = np.abs(np.real(vector))
	
	return(np.reshape(vector, (asset_grid.shape[0], income_states.shape[0]), order = "f"))

start = time.time()
r_star = sp.optimize.newton(lambda x: market_clearing(x)[0]**2, 0.025, tol = 1e-5) # 19 seconds
end = time.time()
print(end - start)


#### output

def gini(x):
	total = 0
	for i, xi in enumerate(x[:-1], 1):
		total += np.nansum(np.abs(xi - x[i:]))
	return total / (len(x)**2 * np.nanmean(x))

def part1_output():
	dist, distribution, K, policy, asset_grid, income_states = market_clearing(r_star)

	wealth_distribution = np.sum(distribution, axis = 1) * asset_grid
	income_distribution = np.sum(distribution, axis = 0) * income_states

	wealth_gini = np.expand_dims(np.around(gini(wealth_distribution), 6), axis = 0)
	income_gini = np.expand_dims(np.around(gini(income_distribution), 6), axis = 0)

	np.savetxt(f'figures/wealth_gini.csv', wealth_gini, delimiter=',')
	np.savetxt(f'figures/income_gini.csv', income_gini, delimiter=',')

	np.percentile(wealth_distribution, 50)
	np.percentile(wealth_distribution, 90)
	np.percentile(wealth_distribution, 99)
	np.percentile(wealth_distribution, 99.9)

	cum_wealth = np.cumsum(wealth_distribution)
	df = pd.DataFrame(cum_wealth, columns = ['Cumulative Wealth'])
	df["Cumulative Wealth"] =  df["Cumulative Wealth"] / cum_wealth[-1]
	df["wealth_level"] = asset_grid
	df["Cumulative Share"] = np.cumsum(np.sum(distribution, axis = 1))
	fig = px.line(df, x="Cumulative Share", y="Cumulative Wealth", title='Lorenz Curve for Wealth')
	fig.write_image(f'figures/lorenz_wealth.png')


	cum_income = np.cumsum(income_distribution)
	df = pd.DataFrame(cum_income, columns = ['Cumulative Income'])
	df["Cumulative Income"] =  df["Cumulative Income"] / cum_income[-1]
	df["income_level"] = income_states
	fig.write_image(f'figures/lorenz_income.png')
	return

part1_output()
#### PART 2
def market_clearing_part2(r_in, max_assets):
min_assets = -1
max_assets = 500
asset_grid = np.logspace(
	start = 0, 
	stop = np.log10(max_assets+2), 
	num = 1501) -2

chi = 0.035/(max_assets - 50)
r_hetero = chi * np.maximum(np.zeros(asset_grid.shape), asset_grid)

r_idio, P_r = tauchenhussey(5, 0,0,0.15, floden_basesigma(0.15, 0))
r_idio = np.exp(r_idio) - 1

alpha = 0.3
sigma = np.sqrt(0.04)
rho = 0.94
mu = 0
inc_size = 5
baseSigma = floden_basesigma(sigma, rho)
log_income_states, transition_matrix = tauchenhussey(inc_size, mu,rho,sigma, baseSigma)
income_states = np.exp(log_income_states[0,:])
steady_distribution = np.linalg.matrix_power(transition_matrix, 1000)
mean_income =  income_states @ steady_distribution[0,:]
income_states = income_states / mean_income # Checked with Tom

delta = 0.1
r_in = 0
K_firm = (alpha/(r_in + delta))**(1/(1-alpha))
wage = (1 - alpha) * K_firm ** alpha
income_states = wage * income_states

params = {
"disc_fact": 0.96,
"risk_aver": 1.5,
"income_states": income_states
}
tol = 1e-3

r = r_in + np.expand_dims(r_hetero, axis = 1) + r_idio
pol_large = egm_coh(income_states, asset_grid, transition_matrix, P_r, r, params, tol)
pol_large_exp = pol_large * np.tile(P_r[0], income_states.shape[0])
# TODO: note, mapping is from exog coh to endog savings, want to force savings to some grid
policy = np.sum(np.reshape(pol_large_exp, (pol_large.shape[0], income_states.shape[0], -1)), axis = 2)
# TODO: continue down here tomorrow:
	policy_exog_grid = griddata(asset_grid, asset_grid, policy, method='nearest')

	ergodic_distribution  = EigenLambda(transition_matrix, asset_grid, income_states, policy_exog_grid)
	
	K_hh = np.sum(ergodic_distribution * policy_exog_grid)
	
	
	diff = K_hh-K_firm
	return(diff, ergodic_distribution, K_hh, policy_exog_grid)


def find_endog_coh(exog_coh, income_states, policy, P, P_r, params):
	''' Returns a mapping from asset tomorrow x income state (cash-on-hand today) to consumption today

	Parameters
	----------
	income_states : numpy array (vector) of persistent income states
	asset_grid : numpy array of possible asset states
	R : numpy array of returns varying with asset level and some idiosyncratic shocks to returns. asset dim x number of idiosyncratic states

	Returns
	-------
	endog_coh
	  The endogenously found cash on hand array.
	'''

	# Find mapping from coh to savings
	# expand to a mapping to coh tomorrow
	# Reduce dimensions again 

		# income 1, all assets | income 2, all assets, etc.
	cons_fut = exog_coh - policy
	mu_cons_fut = cons_fut**(-params["risk_aver"])
	PP = np.kron(P, P_r)
	Ecf = np.matmul(mu_cons_fut, PP.T)
	# Ecf = Ecf * np.tile(P_r, 5)[0]
	cons_today = (params["disc_fact"] * Ecf)**(-1/params["risk_aver"])
	
	savings = exog_coh - cons_today
	# Notice depedence on next period interest rate!!!
	return(savings) # 1501 x 25

def invert_function(exog_grid, x_grid, outcome):
	''' Reverts a mapping from x_grid to endog_outcome to endog outcome to exog_grid.
	This is done using interpolation.
	'''
	policy = np.empty(outcome.shape)
	
	for s in range(outcome.shape[1]):
		policy[:,s] = np.interp(x = exog_grid[:,s], xp = outcome[:,s], fp = x_grid[:,s], left = x_grid[0,s], right = x_grid[-1,s])
	return(policy)

def egm_coh(income_states, asset_grid, P, P_r, returns_matrix, params, tol):
	''' Returns a mapping from endogenous cash on hand X income state to assets tomorrow.

	Parameters
	----------
	income_states : numpy array (vector) of persistent income states
	asset_grid : numpy array of possible asset states
	R : numpy array of returns varying with asset level and some idiosyncratic shocks to returns. asset dim x number of idiosyncratic states

	Returns
	-------
	policy
	  A mapping from cash on hand x income states to assets tomorrow
	'''
	
	
	asset_income_grid = np.logspace(
		start = 0, 
		stop = np.log10(np.max(asset_grid) + 1 - (-1)),
		num = 1501) - 1 + (-1)

	#asset_income_grid = np.tile(asset_income_grid.T, 5).reshape(5, 1501).T
	asset_incomes = (1+returns_matrix) *  np.expand_dims(asset_income_grid,1)
	asset_incomes_rep = np.tile(asset_incomes, income_states.shape[0])
	# All possible cash-on-hand states:
	exog_coh = asset_incomes_rep + np.repeat(income_states, asset_incomes.shape[1])
	# Policy contains all possible future values, i.e. 1501 by 25
	policy_guess = np.copy(exog_coh) - 0.01

	diff = tol + 1
	while diff > tol:
		diff = 0
		savings = find_endog_coh(exog_coh, income_states, policy_guess, P, P_r,params) # given that we know the shock, what was the chocie of coh?
	#	policy_guess_upd = invert_function(exog_grid = exog_coh, x_grid = exog_coh, outcome = savings)
		policy_guess_upd = savings	
		diff = max(np.max(np.abs(policy_guess_upd - policy_guess)), diff)
		policy_guess = np.copy(policy_guess_upd)

	policy = policy_guess
	return(policy)
















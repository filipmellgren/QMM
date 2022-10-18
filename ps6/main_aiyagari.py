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
	return(diff, ergodic_distribution, K_hh, policy_exog_grid)

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
r_star = sp.optimize.bisect(lambda x: market_clearing(x)[0], 0, 0.2) 
end = time.time()
print(end - start)

dist, distribution, K, policy = market_clearing(r_star)



#### output
def gini(x):
	total = 0
	for i, xi in enumerate(x[:-1], 1):
		total += np.nansum(np.abs(xi - x[i:]))
	return total / (len(x)**2 * np.nanmean(x))

wealth_distribution = np.sum(distribution, axis = 1)
income_distribution = np.sum(distribution, axis = 0)

wealth_gini = np.expand_dims(np.around(gini(wealth_distribution), 6), axis = 0)
income_gini = np.expand_dims(np.around(gini(income_distribution), 6), axis = 0)

np.savetxt(f'figures/wealth_gini.csv', wealth_gini, delimiter=',')
np.savetxt(f'figures/income_gini.csv', income_gini, delimiter=',')

np.percentile(wealth_distribution, 50)
np.percentile(wealth_distribution, 0.9)
np.percentile(wealth_distribution, 0.99)
np.percentile(wealth_distribution, 0.999)

cum_wealth = np.cumsum(wealth_distribution)
df = pd.DataFrame(cum_wealth, columns = ['cum wealth'])
df["cum wealth"] =  df["cum wealth"] / cum_wealth[-1]
fig = px.line(df, x=df.index/len(df.index), y="cum wealth", title='Lorenz curve wealth')
fig.write_image(f'figures/lorenz_wealth.png')


cum_income = np.cumsum(income_distribution)
df = pd.DataFrame(cum_income, columns = ['cum income'])
df["cum income"] =  df["cum income"] / cum_income[-1]
fig = px.line(df, x=df.index/len(df.index), y="cum income", title='Lorenz curve income')
fig.write_image(f'figures/lorenz_income.png')













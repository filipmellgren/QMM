
import numpy as np
from household_problem import solve_hh, policy_to_grid, egm_update
from distribution_funcs import get_transition_matrix, get_distribution_fast
from mit_shock import solve_trans_path
from market_class import Market

from numba import jit, prange
import numba as nb
import quantecon as qe
from quantecon import MarkovChain
from quantecon.markov import DiscreteDP
import ipdb
import plotly.express as px
import plotly.graph_objects as go
import time
from scipy.optimize import minimize_scalar
from scipy import optimize
import scipy as sp
from scipy import linalg
from scipy.interpolate import griddata



import warnings
#warnings.filterwarnings("error")

def objective_function(rate_guess, market_object):
	
	market_object.set_prices(rate_guess)

	policy = solve_hh(market_object.P, rate_guess, market_object.wage, market_object.tax, market_object.L_tilde, market_object.mu, market_object.gamma, market_object.beta,market_object.delta, market_object.state_grid, market_object.asset_states)

	policy_ix_up, alpha_list = policy_to_grid(policy, market_object.asset_states)
	
	P = get_transition_matrix(market_object.Q, nb.typed.List(policy_ix_up), nb.typed.List(alpha_list), market_object.state_grid)
	P = np.asarray(P) # P.shape = (2000, 2000) = (#states X #states)

	assert np.all(np.isclose(np.sum(P, axis = 1), 1)), "P is not a transition matrix"
	
	distr_guess = np.full(market_object.state_grid.shape, 1)/len(market_object.state_grid)
	distr = get_distribution_fast(distr_guess, P, market_object.state_grid)

	alpha_array = np.asarray(alpha_list)
	policy_ix_up_array = np.asarray(policy_ix_up)
	savings = market_object.asset_states[policy_ix_up_array] * alpha_array + market_object.asset_states[policy_ix_up_array - 1] * (1 - alpha_array)

	K_HH =  distr @ savings #/ (1+ rate_guess - market_object.delta)

	loss = (market_object.K - K_HH)**2
	print(market_object.K - K_HH)
	return(loss)

def sim_hh_panel(P, T, N_hh):
	''' Simulate a N_hh by T array of hh employment status
	Needs the quantecon package installed (not on python 3.11)
	'''
	mc = qe.MarkovChain(P, state_values = ('unemployed', 'employed'))
	hh_list = []
	for hh in range(N_hh):
		hh_trajectory = mc.simulate_indices(ts_length=T)
		hh_list.append(hh_trajectory)
	return(np.asarray(hh_list))

P = np.array([[0.47, 0.53], [0.04, 0.96]])
hh_panel = sim_hh_panel(P, 1000, 2000) # TODO: might not be strictly necessary
	
steady_state = Market(
	hh_panel,
	r = 0.02,
	gamma = 2, # CRRA parameter
	beta = 0.99, # Discount factor
	P = np.array([[0.47, 0.53], [0.04, 0.96]]), # Employment Markov transition matrix
	mu = 0.15, # Unemployment replacement rate
	delta = 0.025, # Capital depreciation rate
	alpha = 0.36, # Capital share
	borrow_constr = 0, # Borrowing constraint
	max_asset = 500,
	T = 1000,
	rho = 0.95,
	sigma = 0.007,
	a_size = 1000) # 1000

sol = minimize_scalar(objective_function, bounds=(0.01, 0.04), method='bounded', args = steady_state)#, options={'xatol': 1e-12, 'maxiter': 500, 'disp': 0})

rate_ss = sol.x
#rate_ss = 0.022589966512929882
print("The interest rate is:")
print(sol.x)
rate_ss = sol.x
#objective_function(sol.x, steady_state)

#rate_ss = 0.034372948575014266 
#rate_ss = 0.034373827436522834 # When dividing with 1 + r - delta
#rate_ss = 0.033694825028656845 # When not forcing assets to one grid point, but two. 
steady_state.set_prices(rate_ss)
steady_state.K

policy_ss = solve_hh(steady_state.P, rate_ss, steady_state.wage, steady_state.tax, steady_state.L_tilde, steady_state.mu, steady_state.gamma, steady_state.beta,steady_state.delta, steady_state.state_grid, steady_state.asset_states)

policy_ix_up, alpha_list = policy_to_grid(policy_ss, steady_state.asset_states) 

P_ss = get_transition_matrix(steady_state.Q, nb.typed.List(policy_ix_up), nb.typed.List(alpha_list), steady_state.state_grid)
P_ss = np.asarray(P_ss)
distr_guess = np.full(steady_state.state_grid.shape, 1)/len(steady_state.state_grid)
distr = get_distribution_fast(distr_guess, P_ss, steady_state.state_grid)

ss = steady_state

#ss.set_tfp(1, 75.63617405185936)
T = 400
K_guess = np.repeat(ss.K, T+1)

policy_ss = np.reshape(np.asarray(policy_ss), (2, 1000), order = "C")
K_sol, K_HH_evol, tfp_seq, T, rate_path, wage_path = solve_trans_path(ss, T, distr, policy_ss, K_guess)


K_sol.shape

import pandas as pd
df = pd.DataFrame(K_HH_evol)

df["iter"] = df.index

df2 = pd.melt(df, id_vars = "iter", var_name = "time", value_name = "K")

fig = px.line(df2, x="time", y="K", color = "iter")
fig.show()





	df3 = pd.DataFrame(tfp_seq)
	df3 = df3.reset_index()
	fig = px.line(df3, x = "index", y = 0)
	fig.show()

	ss.K


	time = np.arange(len(tmp))
	df = pd.DataFrame(np.array([tmp, time]))



	tmp0 = tmp
	print((1, 2))

	tmp = [1,2,3,4]
	tmp[-1]
	tmp.append(5)

	# Use the Impulse Response Function as a numerically computed derivative

	# Treat the value of a variable at point t as the sum of responses to all past shocks

	tmp = np.linspace(0,10,10)
	np.insert(tmp, 0, 1000)

get_distribution(P)






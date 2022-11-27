
import numpy as np
from household_problem import solve_hh, policy_to_grid, egm_update
from distribution_funcs import get_transition_matrix, get_distribution_fast, get_distribution
from mit_shock import solve_trans_path, get_jacobians
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

	policy = solve_hh(market_object.P, rate_guess, market_object.wage, market_object.tax, market_object.L_tilde, market_object.mu, market_object.gamma, market_object.beta, market_object.delta, market_object.state_grid, market_object.asset_states)

	policy_ix_up, alpha_list = policy_to_grid(policy, market_object.asset_states)
	
	P = get_transition_matrix(market_object.Q, nb.typed.List(policy_ix_up), nb.typed.List(alpha_list), market_object.state_grid)
	P = np.asarray(P) # P.shape = (2000, 2000) = (#states X #states)

	assert np.all(np.isclose(np.sum(P, axis = 1), 1)), "P is not a transition matrix"
	distr = get_distribution(P) # Sometimes, fails when matrix becomes "singular". However, seems stable. The other distribution finder needs a GOOD guess.
	#distr = get_distribution_fast(distr_guess, P, market_object.state_grid) # TODO: this fails when initial guess is bad

	K_HH =  distr @ policy

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

steady_state = Market(
	r = 0.035,
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

sol = minimize_scalar(objective_function, bounds=(0.03, 0.04), method='bounded', args = steady_state)

rate_ss = sol.x 
#rate_ss = 0.03494711589289896

steady_state.set_prices(rate_ss)

policy_ss = solve_hh(steady_state.P, rate_ss, steady_state.wage, steady_state.tax, steady_state.L_tilde, steady_state.mu, steady_state.gamma, steady_state.beta,steady_state.delta, steady_state.state_grid, steady_state.asset_states)

policy_ix_up, alpha_list = policy_to_grid(policy_ss, steady_state.asset_states) 

P_ss = get_transition_matrix(steady_state.Q, nb.typed.List(policy_ix_up), nb.typed.List(alpha_list), steady_state.state_grid)
P_ss = np.asarray(P_ss)

distr_ss = get_distribution(P_ss)
#distr = get_distribution_fast(distr_guess, P_ss, steady_state.state_grid)

ss = steady_state

# MIT Shocks, BKM #####################
T = 150
shock = 0.05


policy_ss = np.reshape(np.asarray(policy_ss), (2, 1000), order = "C")
#K_sol, K_HH_evol, tfp_seq, T, rate_path, wage_path, policy, distr = solve_trans_path(ss, T, distr, policy_ss, tfp, K_guess)

def transpath(shock, T, ss, distr, policy_ss):

	tfp =  1 + shock * 0.95**np.linspace(0,T, T) 
	tfp = np.insert(tfp, 0, 1)
	tfp[-1] = 1
	K_guess = np.repeat(ss.K, T+1) * tfp

	K_sol, K_HH_evol, tfp_seq, T, rate_path, wage_path, policy, distr = solve_trans_path(ss, T, distr, policy_ss, tfp, K_guess)

	return(K_sol, K_HH_evol, tfp_seq, T, rate_path, wage_path, policy, distr)

K_sol, K_HH_evol, tfp_seq, T, rate_path, wage_path, policy, distr = transpath(shock, T, ss, distr_ss, policy_ss)

GDP = tfp_seq * K_sol**(ss.alpha)
C = GDP[:-1] + K_sol[1:] - K_sol[:-1] * (1 + np.asarray(rate_path)[:-1] - ss.delta)


def corr_var(a, b):
	covmat = np.cov(a, b)
	vara = covmat[0,0]
	corr = covmat[0,1]/(np.sqrt(covmat[0,0] * covmat[1,1]))
	return(corr, vara)

Y_var = corr_var(GDP, tfp_seq)
K_var = corr_var(K_sol, tfp_seq)
C_var = corr_var(C, tfp_seq[:-1])

varmat = np.around(np.asarray([Y_var, K_var, C_var]), 4)
np.savetxt("figures/varmat.txt", varmat)





# Auclert ########################
F, J = get_jacobians(150, P_ss, policy_ss, distr, policy, h = 0.001)

# Plots ##########################
import pandas as pd

df = pd.DataFrame(K_HH_evol)

df["iter"] = df.index

df2 = pd.melt(df, id_vars = "iter", var_name = "time", value_name = "K")
df2["iter"] = df2["iter"].astype(float)
time_array = np.linspace(start = 0, stop = len(tfp_seq)-2, num = len(tfp_seq)-1).astype("int")


from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

fig = make_subplots(rows=3, cols=1, subplot_titles=("Hosehold saving training dynamics", "TFP shock path", "Interest path"), shared_xaxes=True)
fig.append_trace(go.Scatter(x = df2["time"], y = df2["K"], mode='markers',
    marker=dict(color=df2["iter"])), row = 1, col = 1 )
fig.append_trace(go.Scatter(x = time_array, y = tfp_seq), row = 2, col = 1)
fig.append_trace(go.Scatter(x = time_array, y = rate_path), row = 3, col = 1)

fig.show()
fig.write_image("figures/training_dynamics.png")

np.min(time_array)

fig1 = px.scatter(df2, x="time", y="K", color = "iter", color_continuous_scale=px.colors.sequential.Viridis, title = "Evolution of Household savings")

fig.show()
fig.write_image("figures/k_path.png")

np.savetxt("figures/K_HH_evol.csv", K_HH_evol, delimiter=",")
np.savetxt("figures/K_sol.csv", K_sol, delimiter=",")
np.savetxt("figures/rate_path.csv", rate_path, delimiter=",")
np.savetxt("figures/wage_path.csv", wage_path, delimiter=",")
np.savetxt("figures/tfp_path.csv", tfp_seq, delimiter=",")

def array_to_plot(array, title_text, path):
	df = pd.DataFrame(array)
	df["time"] = df.index
	fig = px.line(df, x = "time", y = 0, title = title_text)
	fig.write_image("figures/" + path)
	return

array_to_plot(rate_path, "Interest evolution", "rate_path.png")
array_to_plot(wage_path, "Wage evolution", "wage_path.png")
array_to_plot(tfp_seq, "TFP evolution", "tfp_path.png")


array_to_plot(distr, "tmp", "tmp.png")
array_to_plot(distr2, "tmp2", "tmp2.png")

np.sum(tmp[0:999])


np.sum(distr[0:999]) + np.sum(distr[1000:1999])
P
distr[1000]
policy_ix_up[0]
policy_ix_up[999]
policy_ix_up[1000]
policy_ix_up[1999]
alpha_list[0]
alpha_list[1000]
alpha_list[999]
alpha_list[1999]


#
tmp = egm_update(policy_ss, ss.P, ss.r, ss.r, ss.wage, ss.wage, ss.tax, ss.L_tilde, ss.mu, ss.gamma, ss.beta, ss.delta, ss.state_grid, ss.asset_states)
policy_ss[(policy_ss - tmp)>0]

np.sum(policy_ss) / np.sum(tmp)

alpha_list[300]
tmp = get_distribution(P)


distr_guess = np.full(2000, 1, dtype = np.float64)/2000

np.sum(distr_guess)




















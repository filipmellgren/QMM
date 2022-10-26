# Problem set 7, Heterogeneity with aggregate shocks (Krusell Smith)

import numpy as np
from numba import jit, njit, prange
import numba as nb
import ipdb
import time
import scipy as sp
from scipy import interpolate
from scipy.interpolate import interpn
import pandas as pd

params = {
	"asset_grid": np.logspace(
	start = 0,
	stop = np.log10(500+1),
	num= 1000)-1,
	"P": np.array([
	[0.525, 0.35, 0.03125, 0.09375], 
	[0.038889, 0.836111, 0.002083, 0.122917],
	[0.09375, 0.03125, 0.291667, 0.583333],
	[0.009115, 0.115885, 0.024306, 0.850694]]),
	"disc_factor": 0.99,
	"risk_aver": 1.5
	}

P = params["P"]


def employment(boom, pop = 1):
	if not boom:
		return(pop - 0.1)
	return(pop - 0.04)

def tax(boom, pop):
	emp = employment(boom)
	return(ue_rr * (pop - emp) / emp)

def wage(boom, K, alpha, pop = 1):
	L = employment(boom, pop)
	MPL = (1-alpha) * (K/L)**alpha
	if not boom:
		return(0.99 * MPL)
	
	return(1.01 * MPL)

def MPK(boom, K, L, alpha):
	if not boom:
		return(0.99 * alpha * (L/K)**(1-alpha))
	return(1.01 * alpha * (L/K)**(1-alpha))

def get_rates(K, L, alpha):
	delta = 0.025
	rates = np.array([MPK(False, K, L, alpha), MPK(True, K, L, alpha)]) - delta
	return(rates)

def get_prices(K):
	wages = np.array([wage(False), wage(True)])

	return(wages, rates)

def get_income_states(K_agg_grid, alpha):
	# K (all income states)
	# TODO: do not forget taxes
	income_states = np.empty(40, dtype=[("income", "double"), ("K_agg", "double"), ("employed", "bool_"), ("boom", "bool_")]) # TODO hardcoded income states
	income_states["K_agg"] = np.repeat(K_agg_grid, 4)
	income_states["employed"] = np.tile((0,1), 20)
	income_states["boom"] = np.tile((1,1,0,0), 10)

	ue_rr = 0.15
	tax = 0.1 # TODO: should calculate for every capital level

	for state in range(income_states.shape[0]):
		if not income_states["employed"][state]:
			income_states["income"][state] = ue_rr
			continue
		boom = income_states["boom"][state]
		K = income_states["K_agg"][state]
		income_states["income"][state] = (1 - tax) * wage(boom, K, alpha)
	
	return(income_states)

def get_transition_matrix(P, K_agg_grid, beliefs):
	'''
	Create the P matrix tiled horizontally and vertically.
	Create an indicator matrix from (aggregate) K to (aggregate) K' based on beliefs
	Return the new transition matrix
	'''
	
	P_tiled = np.tile(P, (K_agg_grid.shape[0], K_agg_grid.shape[0]))
	K_trans_indic = find_K_transitions(beliefs, K_agg_grid)
	K_trans_indic = np.repeat(np.repeat(K_trans_indic, P.shape[0], axis = 0), P.shape[0], axis = 1)
	PP = P_tiled * K_trans_indic
	return(PP)

def find_K_transitions(beliefs, K_agg_grid):
	
	i_mat = np.empty((K_agg_grid.shape[0], K_agg_grid.shape[0]), dtype = "int8")
	for ix in range(K_agg_grid.shape[0]):
		K = K_agg_grid[ix]
		K_next = get_K_next(beliefs, K)
		K_next_on_grid = K_agg_grid[np.abs(K_agg_grid - K_next).argmin()]
		indicator = np.isclose(K_next_on_grid, K_agg_grid)
		i_mat[ix, :] = indicator
	assert np.all(np.sum(i_mat, axis = 1) == 1), "K indicator not pointing to unique elements."
	return(i_mat)
	
def get_K_next(beliefs, K):
	K_next = np.exp(beliefs[0] + beliefs[1] * K)
	return(K_next)

def interest_rates(K_grid, L, alpha):
	# TODO: is this defined well?
	rates = np.empty((K_grid.shape[0], 4)) # 4 is just idio x agg shocks
	for ix in range(K_grid.shape[0]):
		K = K_grid[ix]
		rate = np.repeat(get_rates(K, L, alpha), 2) # bad, bad, good, good
		rates[ix, :]
	return(rates)

def simulate_shocks(N_hh, T):
	p_good = 0.875
	agg_shock = []
	for time in range(T):
		boom = np.random.uniform() < p_good
		agg_shock.append(boom)

	panel = np.empty((T, N_hh))
	for hh in range(N_hh):
		hh_draws = np.random.uniform(size = T-1)
		hh_employed = sim_hh(T, hh_draws)
		panel[:,hh] = np.asarray(hh_employed)

	return(panel, agg_shock)

@jit(nopython=True, parallel = False)
def sim_hh(T, hh_draws):
	hh_employed= [True]
	for time in prange(T-1):
		p_ee = 0.5 # TODO. Might depend on aggregate state.  
		p_ue = 0.2 # TODO
		hh_draw = hh_draws[time] 
		if hh_employed[-1]:
			if hh_draw > p_ee:
				hh_employed.append(True)
			else:
				hh_employed.append(False)
		else:
			if hh_draw > p_ue:
				hh_employed.append(True)
			else:
				hh_employed.append(False)
	return(hh_employed)

def egm_asset_choices(rate, P, action_states, income_states, policy, disc_factor, risk_aver):
	mu_cons_fut = ((1+rate) * np.expand_dims(action_states, axis = 1)  + income_states - policy)**(-risk_aver)
	Ecf = np.matmul(mu_cons_fut, P.T)
	cons_today = (disc_factor * (1+rate) * Ecf)**(-1/risk_aver)
	endog_assets = 1/(1+rate) * (cons_today + np.expand_dims(action_states, axis = 1) - income_states)
	return(endog_assets)

def interpolate_to_grid(exog_grid, x_grid, outcome):
	
	policy = np.empty(outcome.shape)
	
	for s in range(outcome.shape[1]):
		policy[:,s] = np.interp(x = exog_grid, xp = x_grid[:,s], fp = outcome[:,s], left = outcome[0,s], right = outcome[-1,s])
	return(policy)

def egm(P, asset_grid, income_states, rate, disc_factor, risk_aver, tol = 1e-6):
	""" Implements the Endogenous Grid Method for solving the household problem.
	
	# TODO: only works for infinitely lived household problems. 

	Useful reference: https://alisdairmckay.com/Notes/HetAgents/EGM.html
	The EGM method solves the household problem faster than VFI by using the HH first order condition to find the policy functions. Attributed to Carroll (2006).

	Roughly the algorithm for updating the diff between policy guess and updated policy is:
		* Calculate future consumption using budget constraint and policy guess
		* Calculate marginal utility of consumption today using the Euler Equation. This requires the expected marginal utility of consumption tomorrow. Solve for consumption today. Easy with CRRA utility (which implementation assumes, not with parameter 1 though, the log utility case, approximate with value close to 1).
		* Above gives mapping from action to assets today. Want to flip this relation, do that using interpolation and evaluate at the exogenous grid. Extrapolate using borrowing constraint and maximum possible asset value (later shouldn't be a problem ideally, just increase upper bound).
		* Update policy and calculate the distance metric, compare and terminate or reiterate.

	Parameters
	----------

	P : a transition matrix represented by a numpy array of probabilities of going from row to col
	asset_grid : a numpy grid of possible actions. Tends to be asset_grid
	income_states : a numpy grid of possible income states
	rate : an interest rate of returns to asset holdings
	disc_factor : rate at which households discount the future
	risk_aver : risk aversion parameter in households CRRA utility.

	Returns
	-------
	policy_guess
		Numpy array of dimensions of asset_grid and income states
	"""
	action_n = asset_grid.shape[0]
	income_n = income_states.shape[0]
	policy_guess = np.full((action_n, income_n), np.min(asset_grid)) # TODO update to 45 degree line
	policy_guess += np.tile(np.linspace(0.001, 0.01, action_n).T, income_n).reshape(income_n, action_n).T
	
	diff = tol + 1
	while diff > tol:
		diff = 0
		endog_assets = egm_asset_choices(rate, P, asset_grid, income_states, policy_guess, disc_factor, risk_aver)
		
		policy_guess_upd = interpolate_to_grid(asset_grid, endog_assets, np.tile(np.expand_dims(asset_grid, axis = 1), income_n)) 
			
		diff = max(np.max(np.abs(policy_guess_upd - policy_guess)), diff)
		policy_guess = np.copy(policy_guess_upd)

	return(policy_guess)

def hh_history_panel(shock_panel, agg_shocks, income_states, asset_grid, policy):
	# for each hh and eyar, find income, assets, savings, consumption
	N_hh = shock_panel.shape[1]
	years = shock_panel.shape[0]
	assets = 0
	panel = np.empty(N_hh * years, dtype=[("year", "int16"), ("hh", "int16"), ("income", "double"), ("assets", "double")])
	panel["year"] = np.tile(range(shock_panel.shape[0]),shock_panel.shape[1])
	panel["hh"] = np.repeat(range(shock_panel.shape[1]),shock_panel.shape[0])
	panel["assets"][panel["year"] == 0] = 0
	K_agg_grid = np.unique(income_states["K_agg"])

	panel_year = panel["year"]
	panel_hh = panel["hh"]

	savings_panel = np.zeros(shock_panel.shape)
	income_panel = np.zeros(shock_panel.shape)
	
	start = time.time()
	savings_panel, income_panel = hh_history_loop(N_hh, K_agg_grid, panel, shock_panel, agg_shocks, income_states, asset_grid, panel_year, panel_hh, policy, savings_panel, income_panel)
	end = time.time()
	print(end - start)
	panel["assets"] = savings_panel.flatten("F")
	panel["income"] = income_panel.flatten("F")
	return(panel)

def hh_history_loop(N_hh, K_agg_grid, panel, shock_panel, agg_shocks, income_states, asset_grid, panel_year, panel_hh, policy, savings_panel, income_panel):
	
	ue_benefit = income_states[income_states["employed"] == False]["income"][0]
	ix_ordered = income_states["income"].argsort()
	income_states = income_states[income_states["employed"]]

	income_states_boom_vec = income_states["boom"]
	income_states_K_agg_vec = income_states["K_agg"]

	savings = np.zeros(N_hh)
	for t in range(shock_panel.shape[0]):
		
		K_agg = np.sum(savings)
		K_agg = K_agg_grid[np.argmin(np.abs(K_agg_grid - K_agg))] # Force to grid
		boom = agg_shocks[t]
		
		employed = shock_panel[t, :]
		if boom:
			income = income_states[(boom) & (income_states_K_agg_vec == K_agg)]
		else:
			income = income_states[(not boom) & (income_states_K_agg_vec == K_agg)]
			
		salary = income["income"][0]
		
		income = employed * salary + (1-employed) * ue_benefit
		
		income_states_ordered = income_states["income"]
		policy_ordered = policy[:,ix_ordered][:,19:-1] # TODO: hacky!

		savings = interpn((asset_grid, income_states_ordered), policy_ordered, (savings, income), bounds_error = False, fill_value = 0)
		savings_panel[t,:] = savings
		income_panel[t,:] = income
	return(savings_panel, income_panel)

def find_eq(beliefs, params):
	T = 6000
	shock_panel, agg_shocks = simulate_shocks(N_hh = 10000, T = T) # Want to use the same shocks when root finding. # T by 10 000 = 6000 * 10 0000
	# Time to solve 10000 by 6000: 279 seconds -> 9.3 seconds -> 5.38 secs
	
	K_ss = 1 # TODO solve for this
	K_grid = np.linspace(
		start = 0.85 * K_ss,
		stop = 1.25 * K_ss,
		num = 10)
	alpha = 0.36
	
	income_states = get_income_states(K_grid, alpha)

	diff = 100
	while diff > 1e-4:
		# TODO: replace with optimization algorithm
		P = get_transition_matrix(params["P"], K_grid, beliefs)
		# TODO: incoporate that interest changes depending on aggregate capital 
		# Just calculate the rate for each K and repeat 4 times
		
		rates = (interest_rates(K_grid, 1, alpha)).flatten()
	
		hh_policy = egm(P, params["asset_grid"], income_states["income"], rates, params["disc_factor"], params["risk_aver"], tol = 1e-6) # 1000 by 40
	
		hh_panel = hh_history_panel(shock_panel, agg_shocks, income_states, params["asset_grid"], hh_policy)

		# hh_distribution = get_hh_distribution(hh_panel) # TODO: redundant?
		# Want 1000 by 40 by T, one distr for each T # applies to line above.
		df = pd.DataFrame(hh_panel)
		boom_years = pd.DataFrame(agg_shocks, columns = ["boom"])
		boom_years["year"] = boom_years.index
		K_implied = df.groupby(["year"]).sum()["assets"].reset_index()
		K_implied = K_implied.join(boom_years)
		K_implied_boom = K_implied_boom[K_implied.boom]
		K_implied_bust = K_implied[~K_implied.boom]
		
		
		boom_coef_new = sp.stats.linregress(K_implied_boom[0:-2], y=K_implied_boom[1:-1])
		bust_coef_new = sp.stats.linregress(K_implied_bust[0:-2], y=K_implied_bust[1:-1])
		diff = np.max(np.abs(boom_coef_new - boom_coef)) # TODO: Check all conditions (also busts)
	return(diff)

find_eq(np.array((0,1)), params)





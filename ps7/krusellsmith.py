# Problem set 7, Heterogeneity with aggregate shocks (Krusell Smith)

import numpy as np
from numba import jit, njit, prange
import numba as nb
import ipdb
import time
import scipy as sp
from scipy import interpolate, stats
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
	# TODO: define this 
	if not boom:
		return(pop - 0.1)
	return(pop - 0.04)

def tax(boom, ue_rr, pop):
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
	tax = 0.1 # TODO: should calculate for every capital level and based on employment

	for state in range(income_states.shape[0]):
		if not income_states["employed"][state]:
			income_states["income"][state] = ue_rr
			continue
		boom = income_states["boom"][state]
		K = income_states["K_agg"][state]
		income_states["income"][state] = (1 - tax) * wage(boom, K, alpha)
	
	return(income_states)

def get_transition_matrix(P, K_agg_grid, beliefs):
	''' Find a Markov probability transition matrix given beliefs

	Step 1: Repeat the P matrix, tiling it horizontally and vertically, once for each aggregate capital state.
	Step 2: Create an indicator matrix from (aggregate) K to (aggregate) K' based on beliefs.
		This step is where HH beliefs come into play. The indicator determines what next level aggregate capital states are possible, and therefore determines the interest rates.
	Step 3: Elementwise multiply the tiled P matrix with the K indicator to 'select' what K's are possible and the probabilities of emplioyment/boom within a given aggrgeate K regime.

	Parameters
	P
		Square numpy array denoting transition from idiosyncratic state & aggregate state today to corresponding state tomorrow.
	Returns
	-------
	PP
		numpy array correspodning to a transition matrix from aggregate capital, and economic state today.
	'''

	#P_tiled = np.tile(P, (K_agg_grid.shape[0], K_agg_grid.shape[0])) # Note this includes idiosyncratic shocks, won't affect prices. 
	K_trans_indic = find_K_transitions(beliefs, K_agg_grid)
	
	n_idio_shocks = 2

	K_trans_indic = np.repeat(np.repeat(K_trans_indic, n_idio_shocks, axis = 0), n_idio_shocks, axis = 1)
	#PP = P_tiled * K_trans_indic


	################
	# Pandas indexing ensures the multiplication does what I want it to
	miindex = pd.MultiIndex.from_product(
    [K_agg_grid, ("bad", "good"), ("ue", "e")])

	K_trans_indic_df = pd.DataFrame(K_trans_indic, index = miindex, columns = miindex)
	# 10 is number of K levels
	P_df = pd.DataFrame(np.tile(P, (10,10)), columns = miindex, index = miindex)

	PP = K_trans_indic_df * P_df 
	return(PP)

def find_K_transitions(beliefs, K_agg_grid):
	
	agg_shocks = (0,1)
	n_reps = len(agg_shocks)
	i_mat_len = n_reps * K_agg_grid.shape[0]

	i_mat = np.empty((i_mat_len, i_mat_len), dtype = "int8")
	K_agg_grid_rep = np.repeat(K_agg_grid, n_reps)

	for ix in range(i_mat_len):
		if ix % n_reps == 0:
			belief = beliefs[0] # Should correspond to bust
		if ix % n_reps == 1:
			belief = beliefs[1] # Should correspond to boom
		
		K = K_agg_grid_rep[ix]
		K_next = get_K_next(belief, K)
		K_next_on_grid = K_agg_grid[np.abs(K_agg_grid - K_next).argmin()]
		indicator = np.isclose(K_next_on_grid, K_agg_grid_rep)
		i_mat[ix, :] = indicator
	assert np.all(np.sum(i_mat, axis = 1) == n_reps), "K indicator not pointing to unique elements."
	return(i_mat)
	
def get_K_next(beliefs, K):
	K_next = np.exp(beliefs[0]) * K**beliefs[1]
	return(K_next)

def interest_rates(K_grid, L, alpha):
	''' Calculates interest rates for all TFP shocks and capital levels. 

	'''
	agg_shocks = (1,0)

	n_K_states = K_grid.shape[0]

	rates = np.empty(K_grid.shape[0]* len(agg_shocks), dtype=[("rate", "double"), ("boom", "bool_"), ("K_agg", "double")]) # 4 is just idio x agg shocks
	
	rates["boom"] =  np.tile((1,0), n_K_states)
	rates["K_agg"] = np.repeat(K_grid, len(agg_shocks))

	for ix in range(K_grid.shape[0]):
		K = K_grid[ix]
		rate = get_rates(K, L, alpha)
		rates["rate"][(~rates["boom"]) & (rates["K_agg"] == K)] = rate[1]
		rates["rate"][(rates["boom"]) & (rates["K_agg"] == K)] = rate[0]
		
	return(rates)

def simulate_shocks(N_hh, T):
	p_good = 0.875
	agg_shock = []
	agg_shock.append(True)

	for time in range(T):
		current = agg_shock[time]
		stay = np.random.uniform() < p_good
		if stay:
			agg_shock.append(current)
		else:
			agg_shock.append(not current)

	panel = np.empty((T, N_hh)) # TODO is this made correctly?
	for hh in range(N_hh):
		hh_draws = np.random.uniform(size = T-1)
		hh_employed = sim_hh(T, hh_draws)
		panel[:,hh] = np.asarray(hh_employed)

	return(panel, agg_shock)

@jit(nopython=True, parallel = False)
def sim_hh(T, hh_draws):
	# TODO look into this
	hh_employed= [True]
	for time in prange(T-1):
		p_ee = 0.5 # TODO. Might depend on aggregate state.  
		p_ue = 0.2 # TODO
		hh_draw = hh_draws[time] # TODO guard clause
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
	""" Implements the Endogenous Grid Method for solving the household problem for all states.
	
	# TODO: only works for infinitely lived household problems. 

	Useful reference: https://alisdairmckay.com/Notes/HetAgents/EGM.html
	The EGM method solves the household problem faster than VFI by using the HH first order condition to find the policy functions. Attributed to Carroll (2006).

	Roughly the algorithm for updating the diff between policy guess and updated policy is:
		* Calculate future consumption using budget constraint and policy guess for all states.
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
	# TODO: do I need to 
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

def hh_history_panel(shock_panel, agg_shocks, df, asset_grid, policy):
	# for each hh and eyar, find income, assets, savings, consumption
	N_hh = shock_panel.shape[1]
	years = shock_panel.shape[0]
	assets = 0
	panel = np.empty(N_hh * years, dtype=[("year", "int16"), ("hh", "int16"), ("income", "double"), ("assets", "double")])
	panel["year"] = np.tile(range(shock_panel.shape[0]),shock_panel.shape[1])
	panel["hh"] = np.repeat(range(shock_panel.shape[1]),shock_panel.shape[0])
	panel["assets"][panel["year"] == 0] = 0

	K_agg_grid = np.unique(df["K_agg"])

	panel_year = panel["year"]
	panel_hh = panel["hh"]

	savings_panel = np.zeros(shock_panel.shape)
	income_panel = np.zeros(shock_panel.shape)
	
	start = time.time()
	savings_panel, income_panel = hh_history_loop(N_hh, K_agg_grid, panel, shock_panel, agg_shocks, df, asset_grid, panel_year, panel_hh, policy, savings_panel, income_panel)
	end = time.time()
	print(end - start)
	panel["assets"] = savings_panel.flatten("F")
	panel["income"] = income_panel.flatten("F")
	return(panel)

def hh_history_loop(N_hh, K_agg_grid, panel, shock_panel, agg_shocks, df, asset_grid, panel_year, panel_hh, policy, savings_panel, income_panel):

	ue_benefit = df[df["employed"] == False]["income"][0]
	#ix_ordered = df["income"].argsort()
	income_states_df = df[df["employed"]]
	income_states = df["income"][19:-1]
	policy = policy[:,19:-1]

	income_states_boom_vec = income_states_df["boom"]
	income_states_K_agg_vec_rep = income_states_df["K_agg"]
	income_states_K_agg_vec = np.unique(income_states_K_agg_vec_rep)

	savings = np.zeros(N_hh)
	boom_incomes = np.asarray(income_states_df[income_states_boom_vec])

	bust_incomes = np.asarray(income_states_df[~income_states_boom_vec])
	
	for t in range(shock_panel.shape[0]):
		
		K_agg = np.sum(savings)
		K_agg = K_agg_grid[np.argmin(np.abs(K_agg_grid - K_agg))] # Force to grid
		boom = agg_shocks[t]
		
		employed = shock_panel[t, :]
		
		if boom:
			income = boom_incomes[income_states_K_agg_vec == K_agg]
		else:
			income = bust_incomes[income_states_K_agg_vec == K_agg]
		
		salary = income[0,0]
		
		income = employed * salary + (1-employed) * ue_benefit
		
		savings = interpn((asset_grid, income_states), policy, (savings, income), bounds_error = False, fill_value = np.min(K_agg)/N_hh)
		savings_panel[t,:] = savings
		income_panel[t,:] = income
	return(savings_panel, income_panel)

def update_beliefs(K, ix):
	burn_in = 1000
	ix = ix[burn_in:-2]
	K_present = np.log(K[ix])
	K_lead = np.log(K[ix+1])
	coef_new = sp.stats.linregress(K_present, y=K_lead)
	coef_new = np.array([coef_new.intercept, coef_new.slope])
	return(coef_new)

def find_eq(beliefs, params):
	T = 6000
	shock_panel, agg_shocks = simulate_shocks(N_hh = 10000, T = T) 
	
	K_ss = 17 # TODO solve for this (this comes from Ali)
	K_grid = np.linspace(
		start = 0.85 * K_ss,
		stop = 1.25 * K_ss,
		num = 10)
	alpha = 0.36
	
	income_states = get_income_states(K_grid, alpha)
	rates = (interest_rates(K_grid, 1, alpha)).flatten()
	income_states = pd.DataFrame(income_states)
	rates = pd.DataFrame(rates)

	df = income_states.merge(rates, 'left', left_on = ("K_agg", "boom"), right_on = ('K_agg', 'boom'))
	
	diff = 100
	while diff > 1e-3:
		P = get_transition_matrix(params["P"], K_grid, beliefs)
		df = df.sort_values(by  = ["income"])
		P = np.asarray(P)[df.index][:,df.index]

		hh_policy = egm(P, params["asset_grid"], np.asarray(df["income"]), np.asarray(df["rate"]), params["disc_factor"], params["risk_aver"], tol = 1e-6) # 1000 by 40
	
		# Debug from here
		hh_panel = hh_history_panel(shock_panel, agg_shocks, df, params["asset_grid"], hh_policy)

		df_hh = pd.DataFrame(hh_panel)
		boom_years = pd.DataFrame(agg_shocks, columns = ["boom"])
		boom_years["year"] = boom_years.index
		K_implied = df_hh.groupby(["year"]).sum()["assets"].reset_index()
		
		K_implied = K_implied.merge(boom_years, left_on = "year", right_on = "year")

		boom_ix = K_implied[K_implied.boom]["assets"].index
		bust_ix = K_implied[~K_implied.boom]["assets"].index
		
		boom_coef_new = update_beliefs(K_implied["assets"], boom_ix)
		bust_coef_new = update_beliefs(K_implied["assets"], bust_ix)

		boom_coef = beliefs[1]
		bust_coef = beliefs[0]

		diff = np.max(np.abs(boom_coef_new - boom_coef)) # TODO: Check all conditions (also busts)

		bust_coef = bust_coef * 0.9 + 0.1 * bust_coef_new
		boom_coef = boom_coef * 0.9 + 0.1 * boom_coef_new

		beliefs = np.array([bust_coef,boom_coef])
		print(diff)
		print(beliefs)
	return(beliefs)

start_belief = np.array([[0.1,1], [0.1,1]])

belief_star = find_eq(start_belief, params)



# TODO LIST
# Calculate tax based on aggregate employment
# Markov structure of income shocks (?) Yes? See slides I guess
# Steady state capital
# How to treat L?
# HH needs to take next period rates into account
# These will depend on capital levels, which are set by the HH beliefs


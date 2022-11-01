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
import itertools

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

#params["asset_grid"] = np.linspace(start = 0, stop = 300, num = 1500)


P = params["P"]



def employment(boom, pop):
	if not boom:
		return(pop*(1-0.1))
	return(pop*(1-0.04))

def tax(boom, ue_rr, pop):
	emp = employment(boom, pop)
	return(ue_rr * (pop - emp) / emp)

def wage(boom, K, alpha, pop):
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
	bust_emp = employment(False, L)
	boom_emp = employment(True, L)
	rates = np.array([MPK(False, K, bust_emp, alpha), MPK(True, K, boom_emp, alpha)]) - delta
	return(rates)

def get_income_states(K_agg_grid, L_tilde, alpha):
	# K (all income states)
	income_states = np.empty(40, dtype=[("income", "double"), ("K_agg", "double"), ("employed", "bool_"), ("boom", "bool_")]) # TODO hardcoded income states
	income_states["K_agg"] = np.repeat(K_agg_grid, 4)
	income_states["employed"] = np.tile((0,1), 20)
	income_states["boom"] = np.tile((1,1,0,0), 10)

	ue_rr = 0.15
	pop = L_tilde

	for state in range(income_states.shape[0]):
		if not income_states["employed"][state]:
			income_states["income"][state] = ue_rr
			continue
		boom = income_states["boom"][state]
		tax_rate = tax(boom, ue_rr, pop)
		K = income_states["K_agg"][state]
		income_states["income"][state] = (1 - tax_rate) * wage(boom, K, alpha, pop)
	
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

def interest_rates(K_grid, L_tilde, alpha):
	''' Calculates interest rates for all TFP shocks and capital levels. 

	'''
	agg_shocks = (1,0)

	L = L_tilde
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

def simulate_shocks(N_hh, T, P):
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
	p_b_ue = np.sum(P[0,(1,3)])
	p_b_e = np.sum(P[1,(1,3)])
	p_g_ue = np.sum(P[2,(1,3)])
	p_g_e = np.sum(P[3,(1,3)])
	# TODO: can this be jitted?
	for hh in range(N_hh):
		hh_draws = np.random.uniform(size = T-1)
		hh_employed = sim_hh(agg_shock, T, hh_draws, p_b_ue, p_b_e, p_g_ue, p_g_e)
		panel[:,hh] = np.asarray(hh_employed)

	return(panel, agg_shock)

@jit(nopython=True, parallel = False)
def sim_hh(agg_shock, T, hh_draws, p_b_ue, p_b_e, p_g_ue, p_g_e):
	hh_employed = [True]
	for time in prange(T-1):
		hh_draw = hh_draws[-1]
		if hh_draw < p_b_ue:
			hh_employed.append(True)
			continue
		if hh_draw > p_g_e:
			hh_employed.append(False)
			continue
		employed = hh_employed[-1]
		if not employed:
			if hh_draw < p_g_ue:
				hh_employed.append(True)
				continue
			if hh_draw > p_g_ue:
				hh_employed.append(False)
				continue
		if hh_draw < p_b_e:
				hh_employed.append(True)
				continue
		boom = agg_shock[time]
		if not boom:
			if hh_draw > p_b_e:
				hh_employed.append(False)
				continue
		if hh_draw < p_g_e:
			hh_employed.append(True)
	return(hh_employed)

def egm_asset_choices(rate, P, action_states, income_states, policy, disc_factor, risk_aver):
	# TODO: unsure this works as intended. Deprecate eventually.
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

def hh_history_panel(shock_panel, agg_shocks, df, asset_grid, policy, income_order):
	# for each hh and eyar, find income, assets, savings, consumption
	N_hh = shock_panel.shape[1]
	years = shock_panel.shape[0]
	panel = np.empty(N_hh * years, dtype=[("year", "int16"), ("hh", "int16"), ("income", "double"), ("assets", "double")])
	panel["year"] = np.tile(range(shock_panel.shape[0]),shock_panel.shape[1])
	panel["hh"] = np.repeat(range(shock_panel.shape[1]),shock_panel.shape[0])

	K_agg_grid = np.unique(df["K_agg"])

	panel_year = panel["year"]
	panel_hh = panel["hh"]

	savings_panel = np.zeros(shock_panel.shape)
	income_panel = np.zeros(shock_panel.shape)
	
	start = time.time()
	savings_panel, income_panel = hh_history_loop(N_hh, K_agg_grid, panel, shock_panel, agg_shocks, df, asset_grid, panel_year, panel_hh, policy, savings_panel, income_panel, income_order)
	end = time.time()
	print(end - start)
	panel["assets"] = savings_panel.flatten("F")
	panel["income"] = income_panel.flatten("F")
	return(panel)

def hh_history_loop(N_hh, K_agg_grid, panel, shock_panel, agg_shocks, df, asset_grid, panel_year, panel_hh, policy, savings_panel, income_panel, income_order):

	ue_benefit = df[df["employed"] == False]["income"][0]
	#ix_ordered = df["income"].argsort()
	income_states_df = df[df["employed"]]
	income_states = df["income"][19:-1]
	income_states = income_order
	income_order_ix = np.argsort(income_states)
	income_states = income_states[income_order_ix]
	policy = policy[:,income_order_ix]
	# TODO: only keep distinct income ix to keep things small
	
	income_states_boom_vec = income_states_df["boom"]
	income_states_K_agg_vec_rep = income_states_df["K_agg"]
	income_states_K_agg_vec = np.unique(income_states_K_agg_vec_rep)

	savings = np.zeros(N_hh) + 17 # TODO: K_ss
	boom_incomes = np.asarray(income_states_df[income_states_boom_vec])

	bust_incomes = np.asarray(income_states_df[~income_states_boom_vec])
	
	for t in range(shock_panel.shape[0]):
		
		K_agg = np.mean(savings)
		K_agg = K_agg_grid[np.argmin(np.abs(K_agg_grid - K_agg))] # Force to grid
		boom = agg_shocks[t]
		
		employed = shock_panel[t, :]
		
		if boom:
			income = boom_incomes[income_states_K_agg_vec == K_agg]
		else:
			income = bust_incomes[income_states_K_agg_vec == K_agg]
		
		salary = income[0,0]
		
		income = employed * salary + (1-employed) * ue_benefit
		
		# TODO: can I redefine the policy matrix to make it easier to index?
		# like policy[agg_shock, K_agg, employment_vec, savings]
		
		income_ix = np.searchsorted(income_states, income)
		asset_ix = np.searchsorted(asset_grid, savings) # previous savings corresponds to current assets
		savings = policy[asset_ix, income_ix]
		
		#savings = interpn((asset_grid, income_states), policy, (savings, income), bounds_error = False, fill_value = np.min(K_agg))

		savings_panel[t,:] = savings
		income_panel[t,:] = income
	return(savings_panel, income_panel)

def update_beliefs(K, ix):
	burn_in = 1000
	ix = ix[burn_in:-2]
	K_present = np.log(K[ix])
	K_lead = np.log(K[ix+1])
	slope, intercept, r, p, se = sp.stats.linregress(K_present, y=K_lead)
	coef_new = np.array([intercept, slope])
	return(coef_new, r**2)

def find_eq(beliefs, params):
	T = 6000
	shock_panel, agg_shocks = simulate_shocks(1000, T, params["P"]) 
	
	K_ss = 40 # TODO solve for this (this comes from Ali)
	K_grid = np.linspace(
		start = 0.75 * K_ss,
		stop = 1.35 * K_ss,
		num = 10)
	alpha = 0.36
	L_tilde = 1/0.9
	
	income_states = get_income_states(K_grid, L_tilde, alpha)
	rates = (interest_rates(K_grid, L_tilde, alpha)).flatten()
	income_states = pd.DataFrame(income_states)
	rates = pd.DataFrame(rates)

	df = income_states.merge(rates, 'left', left_on = ("K_agg", "boom"), right_on = ('K_agg', 'boom'))
	
	loss = 100

	while loss > 1e-3:
		P = get_transition_matrix(params["P"], K_grid, beliefs)
		income_index = df.sort_values(by  = ["income"]).index

		#P = np.asarray(P)[df.index][:,df.index]
		# TODO: current policy gives first guess. Why does it not update better?
		hh_policy, income_order = egm_dataframe(df, P, K_grid, params["asset_grid"], params, tol = 1e-6)
		#hh_policy = egm(P, params["asset_grid"], np.asarray(df["income"]), np.asarray(df["rate"]), params["disc_factor"], params["risk_aver"], tol = 1e-6) # 1000 by 40
		ipdb.set_trace()
		# Debug this
		
		hh_panel = hh_history_panel(shock_panel, agg_shocks, df, params["asset_grid"], hh_policy, income_order)
		# Debug above

		df_hh = pd.DataFrame(hh_panel)
		boom_years = pd.DataFrame(agg_shocks, columns = ["boom"])
		boom_years["year"] = boom_years.index
		K_implied = df_hh.groupby(["year"]).mean()["assets"].reset_index()
		
		K_implied = K_implied.merge(boom_years, left_on = "year", right_on = "year")

		boom_ix = K_implied[K_implied.boom]["assets"].index
		bust_ix = K_implied[~K_implied.boom]["assets"].index
		
		boom_coef_new, r2 = update_beliefs(K_implied["assets"], boom_ix)
		bust_coef_new, r2 = update_beliefs(K_implied["assets"], bust_ix)

		bust_coef = beliefs[0]
		boom_coef = beliefs[1]

		diff = np.max(boom_coef_new - boom_coef)
		loss = diff**2

		bust_coef = bust_coef * 0.9 + 0.1 * bust_coef_new
		boom_coef = boom_coef * 0.9 + 0.1 * boom_coef_new

		beliefs = np.array([bust_coef,boom_coef])
		print(loss)
		print(beliefs)
		print(r2)
	return(beliefs)

def egm_dataframe(df, P, K_grid, asset_grid, params, tol = 1e-6):
	df_copy = df.copy()

	action_n = params["asset_grid"].shape[0]
	income_n = 40 #income_states.shape[0]

	assets_df = itertools.product(asset_grid, K_grid, [True, False])
	assets_df = pd.DataFrame(assets_df)
	assets_df = assets_df.rename(columns={0: "assets", 1: "K_agg", 2 : "boom"})
	df = df.merge(assets_df, how = "left", left_on=["K_agg", "boom"], right_on = ["K_agg", "boom"]) # All possible states
	df = df.set_index(["K_agg", "employed", "boom", "assets"])

	policy_guess = df.copy()
	
	policy_guess["policy_guess"] = policy_guess.index.get_level_values(3)# * 0.98 # Guess must be on grid
	df["policy_guess"] = policy_guess.index.get_level_values(3)
	
	# pol guess to something we can join to df
	
	diff = tol + 1

	P_df = get_P_df(P)
	while diff > tol:
		print(diff)
		diff = 0
		
		policy_guess_mat, income_order, policy_guess_upd = egm_policy(df, P_df, K_grid, policy_guess["policy_guess"], params)
		
		policy_guess_upd = policy_guess_upd.reset_index().set_index(["K_agg", "boom", "employed", "assets"])

		if policy_guess_upd["policy_guess"].isnull().values.any():
			ipdb.set_trace()
			pass
		
		diff = max(np.max(np.abs(policy_guess_upd["policy_guess"] - policy_guess["policy_guess"])), diff)
		diff = max(policy_guess_upd["policy_guess"] - policy_guess["policy_guess"])
		policy_guess = policy_guess_upd.copy()

	return(policy_guess_mat, income_order)

def get_P_df(P):
	P.index = P.index.set_names(["K_agg_f", "boom_f", "employed_f"])
	P_df = pd.melt(P, id_vars=None, value_vars=None, var_name=["K_agg", "boom", "employed"], value_name='prob', col_level=None, ignore_index=False).reset_index()

	P_df.loc[P_df["boom"] == "good", "boom"] = True
	P_df.loc[P_df["boom"] == "bad", "boom"] = False
	P_df.loc[P_df["boom_f"] == "good", "boom_f"] = True
	P_df.loc[P_df["boom_f"] == "bad", "boom_f"] = False

	P_df.loc[P_df["employed"] == "e", "employed"] = True
	P_df.loc[P_df["employed"] == "ue", "employed"] = False
	P_df.loc[P_df["employed_f"] == "e", "employed_f"] = True
	P_df.loc[P_df["employed_f"] == "ue", "employed_f"] = False
	return(P_df)

def get_df_future(df, P_df, params):
	''' Gives df with all possible values of mu_cons, with associated transition probabilities form all possible states
	'''
	
	df["mu_cons"] = ((1+df.rate) * df.assets + df.income - df.policy_guess)**(-params["risk_aver"])
	rate_today = df[["K_agg","boom", "rate"]].drop_duplicates().rename(columns = {"K_agg" : "K_agg_f", "boom" : "boom_f"}).set_index(["K_agg_f", "boom_f"])
	income_today = df[["K_agg","boom", "employed", "income"]].drop_duplicates().rename(columns = {"K_agg" : "K_agg_f", "boom" : "boom_f", "employed" : "employed_f"}).set_index(["K_agg_f", "boom_f", "employed_f"])
	
	if np.any((1+df.rate) * df.assets + df.income - df.policy_guess < 0):
		print(min((1+df.rate) * df.assets + df.income - df.policy_guess))
		ipdb.set_trace()
	if df.mu_cons.isnull().sum() != 0:
		ipdb.set_trace()
		
		tmp = df[(1+df.rate) * df.assets + df.income - df.policy_guess < 0]
		
	assert df.mu_cons.isnull().sum() == 0, "There is negative consumption given policy."

	df_fut = df[["K_agg", "boom", "employed", "assets", "mu_cons", "rate"]]
	df_fut = df_fut.merge(P_df, how = "left", left_on = ["K_agg", "boom", "employed"], right_on = ["K_agg", "boom", "employed"])
	df_fut["exp_term"] = df_fut["prob"] * df_fut["mu_cons"] * (1 + df_fut["rate"])
	# Exclude values made impossible by the K law of motion
	ipdb.set_trace()

	df_fut = df_fut[df_fut["exp_term"]!=0]
	df_fut = df_fut.set_index(["K_agg_f", "boom_f", "employed_f", "assets"])
	df_fut = df_fut[["exp_term", "prob"]]
	
	df = df_fut.groupby(level = [0,1,2,3]).sum()
	disc_factor = 0.99
	risk_aver = 1.5

	# Rate already in expectation
	df["cons_today"] = (disc_factor * df["exp_term"])**(-1/risk_aver)
	
	df = df.join(rate_today)
	df = df.join(income_today)
	df = df.reset_index()
	df["endog_assets"] = (1/(1+df["rate"])) * (df["cons_today"] + df["assets"] - df["income"])
	df = df.rename(columns = {"K_agg_f" : "K_agg", "boom_f" : "boom", "employed_f" : "employed"})

	assert df_fut.exp_term.isnull().sum() == 0, "There are null values in the expectation term of the df_fut DataFrame."
	assert min(exp_df["prob"]) == 1, "Probabilities sum to less than one"
	assert max(exp_df["prob"]) == 1, "Probabilities sum to more than one"
	return(df)

def egm_policy(df, P_df, K_grid, policy_guess, params):

	disc_factor = 0.99
	risk_aver = 1.5
	asset_grid_n = params["asset_grid"].shape[0]
	
	df = df.reset_index()
	df["mu_cons"] = ((1+df.rate) * df.assets + df.income - df.policy_guess)**(-params["risk_aver"])

	rate_today = df[["K_agg","boom", "rate"]].drop_duplicates().rename(columns = {"K_agg" : "K_agg_f", "boom" : "boom_f"}).set_index(["K_agg_f", "boom_f"])
	income_today = df[["K_agg","boom", "employed", "income"]].drop_duplicates().rename(columns = {"K_agg" : "K_agg_f", "boom" : "boom_f", "employed" : "employed_f"}).set_index(["K_agg_f", "boom_f", "employed_f"])
	
	if np.any((1+df.rate) * df.assets + df.income - df.policy_guess < 0):
		print(min((1+df.rate) * df.assets + df.income - df.policy_guess))
		ipdb.set_trace()
	if df.mu_cons.isnull().sum() != 0:
		ipdb.set_trace()
		tmp = df[(1+df.rate) * df.assets + df.income - df.policy_guess < 0]
		
	assert df.mu_cons.isnull().sum() == 0, "There is negative consumption given policy."

	df_fut = df[["K_agg", "boom", "employed", "assets", "mu_cons", "rate"]]
	df_fut = df_fut.merge(P_df, how = "left", left_on = ["K_agg", "boom", "employed"], right_on = ["K_agg", "boom", "employed"])
	df_fut["exp_term"] = df_fut["prob"] * df_fut["mu_cons"] * (1 + df_fut["rate"])
	# Exclude values made impossible by the K law of motion
	
	df_fut = df_fut[df_fut["exp_term"]!=0]
	df_fut = df_fut.set_index(["K_agg_f", "boom_f", "employed_f", "assets"])
	df_fut = df_fut[["exp_term", "prob"]]
	
	df = df_fut.groupby(level = [0,1,2,3]).sum()
	
	# Rate already in expectation
	df["cons_today"] = (disc_factor * df["exp_term"])**(-1/risk_aver)
	
	df = df.join(rate_today)
	df = df.join(income_today)
	df = df.reset_index()
	df["endog_assets"] = (1/(1+df["rate"])) * (df["cons_today"] + df["assets"] - df["income"])
	df = df.rename(columns = {"K_agg_f" : "K_agg", "boom_f" : "boom", "employed_f" : "employed"})

	assert df_fut.exp_term.isnull().sum() == 0, "There are null values in the expectation term of the df_fut DataFrame."
	assert min(df["prob"]) == 1, "Probabilities sum to less than one"
	assert max(df["prob"]) == 1, "Probabilities sum to more than one"
	
	df = df.set_index(["K_agg", "boom", "employed", "assets"])

	policy_mat = np.empty((asset_grid_n, 40)) # TODO don't hard code
	exog_grid = params["asset_grid"]
	ss = 0
	income_order = np.empty(40)
	ix_list = []
	policy_list = []
	
	for ix, new_df in df.groupby(level=[0,1,2]):
		
		ix_list.append(ix)
		income = new_df.income.iloc[0]
		income_order[ss] = income
		endog_x = np.asarray(new_df["endog_assets"])
		outcome = np.asarray(new_df.index.get_level_values(3))
		
		policy_exog_grid = np.interp(x = exog_grid, xp = endog_x, fp = outcome)
		
		#index_val = np.searchsorted(exog_grid, policy_exog_grid, side = "left")
		# TODO: make work without this. Might kill idnividuals by rounding error!
		# Maybe pass on grid onto next iteration to still be able to do the join
		policy_mat[:,ss] = policy_exog_grid 
		#policy_mat[:,ss] = exog_grid[index_val] # round to grid for compatibility with later joins
		ss += 1
		
	new_policy_guess = pd.DataFrame(policy_mat, columns = ix_list)
	new_policy_guess["assets"] = exog_grid
	new_policy_guess = pd.melt(new_policy_guess, id_vars = "assets")
	new_policy_guess.index = pd.MultiIndex.from_tuples(new_policy_guess.variable)
	new_policy_guess = new_policy_guess.rename(columns = {"value" : "policy_guess"})
	new_policy_guess.index = new_policy_guess.index.set_names(["K_agg", "boom", "employed"])
	new_policy_guess = new_policy_guess[["policy_guess", "assets"]]

	return(policy_mat, income_order, new_policy_guess)


start_belief = np.array([[0,1], [0,1]])
#start_belief = np.array([[0.2,0.9], [0.2,0.9]])

belief_star = find_eq(start_belief, params)

# TODO LIST
# Don't kill people by rounding

# Steady state capital

# HH needs to take next period rates into account
# These will depend on capital levels, which are set by the HH beliefs


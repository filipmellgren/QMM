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

K_ss =32 # Aprox from Solow model assuming s = 0.1, r-delta = 0.01.
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
	"risk_aver": 1.5,
	"ue_rr": 0.15,
	"delta" : 0.025,
	"K_grid" : np.linspace(
		start = 0.85 * K_ss,
		stop = 1.25 *  K_ss,
		num = 10),
	"alpha" : 0.36,
	"L_tilde" : 1/0.9,
	"T" : 6000,
	"Nhh" : 10000
	}

P = params["P"]

def employment(boom, pop):
	return(boom * pop * (1-0.04) + (1 - boom) * pop * (1 - 0.1))

def wage(boom, K, alpha, pop):
	L = employment(boom, pop)
	z = boom * 1.01 + (1 - boom) * 0.99
	MPL = z * (1-alpha) * (K/L)**alpha
	return(MPL)

def MPK(boom, K, L, alpha):
	z = boom * 1.01 + (1 - boom) * 0.99
	MPK = z * alpha * (L/K)**(1-alpha)
	return(MPK)

def get_income_states(K_agg_grid, L_tilde, alpha):
	df = np.empty(40, dtype=[("income", "double"), ("K_agg", "double"), ("employed", "bool_"), ("boom", "bool_")])
	df = pd.DataFrame(df)
	df["K_agg"] = np.repeat(K_agg_grid, 4)
	df["employed"] = np.tile((0,1), 20)
	df["boom"] = np.tile((1,1,0,0), 10)

	ue_rr = params["ue_rr"]
	pop = L_tilde
	emp = employment(df["boom"], pop)
	tax_rate = ue_rr * (pop - emp) / (emp)

	df["wage"] = wage(df["boom"], df["K_agg"], alpha, pop)
	df["income"] = df["employed"] * (1 - tax_rate) * df["wage"] + (1-df["employed"]) * df["wage"] * ue_rr
	return(df)

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
	
	# Boom
	belief = beliefs[1]
	df_boom = find_K_transitions(belief, K_agg_grid)
	df_boom["boom"] = True

	# Bust
	belief = beliefs[0]
	df_bust = find_K_transitions(belief, K_agg_grid)
	df_bust["boom"] = False

	df_k_trans = pd.concat([df_bust, df_boom])
	df_k_trans.K_to = df_k_trans.K_to.astype("float64")
	df_k_trans = df_k_trans.set_index(["K_from", "boom"])

	miindex = pd.MultiIndex.from_product(
    [("False", "True"), ("False", "True")])

	P_df = pd.DataFrame(P, columns = miindex)
	P_df["employed"] = np.array([False,True,False,True])
	P_df["boom"] = np.array([False,False,True,True])

	P_df = P_df.melt(id_vars = ["employed", "boom"], var_name = ["boom_next","employed_next"], value_name = "prob").set_index(["employed", "boom"])

	P_df.boom_next = P_df.boom_next.replace({"True": True, "False": False})
	P_df.employed_next = P_df.employed_next.replace({"True": True, "False": False})
	
	trans_df = P_df.join(df_k_trans)

	trans_df["prob"] = trans_df["prob"] * trans_df["trans"]

	assert np.all(np.isclose(trans_df.groupby(level=[0,1,2]).sum()["prob"], 1)), "Probabilities don't sum to one"
	return(trans_df)

def find_K_transitions(belief, K_agg_grid):
	
	K_next = get_K_next(belief, K_agg_grid)
	grid_index = np.maximum(np.searchsorted(K_agg_grid, K_next)-1, 0) # TODO: Could searchsorted be rounding in bad ways?
	K_next_grid = K_agg_grid[grid_index]
	K_now, K_next = np.meshgrid(K_agg_grid, K_next_grid, sparse = True, indexing = "ij")
	i_mat = K_now == K_next
	i_mat = i_mat.T
	assert np.all(np.sum(i_mat, axis = 1) == 1), "K indicator not pointing to unique elements."
	k_index = pd.Index(K_agg_grid)

	df = pd.DataFrame(i_mat, index = k_index, columns = k_index)
	df["K_from"] = df.index
	df = df.melt(id_vars = "K_from", var_name = "K_to", value_name = "trans")
	return(df)
	
def get_K_next(beliefs, K):
	K_next = np.exp(beliefs[0] + beliefs[1] * np.log(K))
	return(K_next)

def get_rates(K_grid, L_tilde, alpha, params):
	''' Calculates interest rates for all TFP shocks and capital levels. 

	'''
	agg_shocks = (1,0)
	n_K_states = K_grid.shape[0]
	delta = params["delta"]

	df = np.empty(K_grid.shape[0]* len(agg_shocks), dtype=[("rate", "double"), ("boom", "bool_"), ("K_agg", "double")]) # 4 is just idio x agg shocks
	df = pd.DataFrame(df)
	df["boom"] =  np.tile((1,0), n_K_states)
	df["K_agg"] = np.repeat(K_grid, len(agg_shocks))
	pop = L_tilde
	emp = employment(df["boom"], pop)
	df["rate"] = MPK(df["boom"], df["K_agg"], emp, alpha) - delta
		
	return(df)

def hh_history_panel(shock_panel, agg_shocks, df, asset_grid, policy, income_order, L_tilde, df_pol):
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
	savings_panel, income_panel = hh_history_loop(N_hh, K_agg_grid, panel, shock_panel, agg_shocks, df, asset_grid, panel_year, panel_hh, policy, savings_panel, income_panel, income_order, L_tilde, df_pol)
	end = time.time()
	print(end - start)
	
	
	assets_df = pd.DataFrame(savings_panel)
	income_df = pd.DataFrame(income_panel)
	assets_df["year"] = assets_df.index
	income_df["year"] = income_df.index

	assets_df = pd.melt(assets_df, id_vars = "year", var_name = "hh", value_name = "savings").set_index(["hh", "year"])
	income_df = pd.melt(income_df, id_vars = "year", var_name = "hh", value_name = "income").set_index(["hh", "year"])
	
	panel = assets_df.join(income_df)

	return(panel)

def hh_history_loop(N_hh, K_agg_grid, panel, shock_panel, agg_shocks, df, asset_grid, panel_year, panel_hh, policy, savings_panel, income_panel, income_order, L_tilde, df_pol):
	boom_incomes = df[(df["boom"] == True) & (df["employed"] == True)]["income"]
	bust_incomes = df[(df["boom"] == False) & (df["employed"] == True)]["income"]
	income_states_K_agg_vec = df["K_agg"].unique()
	savings = np.zeros(N_hh) + np.mean(income_states_K_agg_vec)

	income_order_ix = np.argsort(income_order)
	income_sorted = np.sort(income_order)
	policy = policy[:,income_order_ix]

	#income_sorted[income_ix]
	#ue_benefit
	#income
	
	for t in range(shock_panel.shape[0]):
		K_agg = np.mean(savings)
		K_agg = K_agg_grid[np.argmin(np.abs(K_agg_grid - K_agg))] # Force to grid
		boom = agg_shocks[t]
		employed = shock_panel[t, :]
		
		incomes = df[(df["boom"] == boom) & (df["K_agg"] == K_agg)]
		
		salary = incomes[incomes["employed"]==1]["income"].iloc[0] # might be same as incomes.iloc[1, 0]
		ue_benefit = incomes[incomes["employed"]==0]["income"].iloc[0] # incomes.iloc[0, 0]

		income = employed * salary * L_tilde + (1-employed) * ue_benefit
		
		income_ix_e = np.argmin(np.abs(salary-income_sorted))
		income_ix_ue = np.argmin(np.abs(ue_benefit-income_sorted)) 
		income_ix = employed * income_ix_e + (1-employed) * income_ix_ue
		income_ix = income_ix.astype("int")
		#income_ix = np.maximum(np.searchsorted(income_order, income)-1, 0)
		asset_ix = np.maximum(np.searchsorted(asset_grid, savings)-1, 0) # previous savings corresponds to current assets

		savings = policy[asset_ix, income_ix]

		savings_panel[t,:] = savings
		income_panel[t,:] = income
	return(savings_panel, income_panel)

@jit(nopython=True, parallel = False)
def simulate_shocks(N_hh, T, P):
	p_change = 0.875
	agg_shock = []
	agg_shock.append(True)

	for time in range(T):
		current = agg_shock[time]
		stay = np.random.rand(1) < p_change
		if stay:
			agg_shock.append(current)
		else:
			agg_shock.append(not current)

	panel = np.empty((T, N_hh))
	
	for hh in range(N_hh):
		hh_draws = np.random.rand(T-1)
		hh_employed = sim_hh(agg_shock, T, hh_draws)
		panel[:,hh] = np.asarray(hh_employed)

	return(panel, agg_shock)

@jit(nopython=True, parallel = False)
def sim_hh(agg_shock, T, hh_draws):
	# Divide in to quadrants
	hh_employed = [True]
	es_boom = 3
	es_bust = 1
	quad1 = np.array([P[0,es_bust]/(0.875), P[1,es_bust]/(0.875)])
	quad2 = np.array([P[0, es_boom]/(1-0.875), P[1, es_boom]/(1-0.875)])
	quad3 = np.array([P[2,es_bust]/(1-0.875), P[3,es_bust]/(1-0.875)])
	quad4 = np.array([P[2,es_boom]/(0.875), P[3,es_boom]/(0.875)])

	for time in prange(T-1):
		boom = agg_shock[time]
		boom_next = agg_shock[time + 1]
		employed = hh_employed[time-1]
		if boom:
			if boom_next:
				if employed:
					p_emp = quad4[1]
				else:
					p_emp = quad4[0]
			else:
				if employed:
					p_emp = quad3[1]
				else:
					p_emp = quad3[0]
		else:
			if boom_next:
				if employed:
					p_emp = quad2[1]
				else:
					p_emp = quad2[0]
			else:
				if employed:
					p_emp = quad1[1]
				else:
					p_emp = quad1[0]

		hh_draw = hh_draws[time-1]
		if hh_draw < p_emp:
			hh_employed.append(True)
			continue
		else:
			hh_employed.append(False)
			continue
	return(hh_employed)

def update_beliefs(K, ix):
	burn_in = 1000
	ix = ix[burn_in:-2]
	K_present = np.log(K[ix])
	K_lead = np.log(K[ix+1])
	slope, intercept, r, p, se = sp.stats.linregress(K_present, y=K_lead)
	coef_new = np.array([intercept, slope])
	return(coef_new, r**2)

def find_eq(beliefs, shock_panel, agg_shocks, df, params):
	
	shock_panel, agg_shocks = simulate_shocks(params["Nhh"], params["T"], params["P"])
		
	df_inc = get_income_states(params["K_grid"], params["L_tilde"], params["alpha"])
	df_rates = get_rates(params["K_grid"], params["L_tilde"], params["alpha"], params)

	df = df_inc.merge(df_rates, 'left', left_on = ("K_agg", "boom"), right_on = ('K_agg', 'boom'))

	loss = 100

	while loss > 1e-5:
		P = get_transition_matrix(params["P"], params["K_grid"], beliefs)
		
		hh_policy, income_order, df_pol = egm_dataframe(df, P, params["K_grid"], params["asset_grid"], params, tol = 1e-6)
		

		hh_panel = hh_history_panel(shock_panel, agg_shocks, df, params["asset_grid"], hh_policy, income_order, params["L_tilde"], df_pol)

		K_implied, boom_ix, bust_ix = get_K_series(hh_panel, agg_shocks)
		
		boom_coef_new, r2 = update_beliefs(K_implied["savings"], boom_ix)
		bust_coef_new, r2 = update_beliefs(K_implied["savings"], bust_ix)

		bust_coef = beliefs[0]
		boom_coef = beliefs[1]
		
		loss = np.sum((boom_coef_new - boom_coef)**2 + (bust_coef_new - bust_coef)**2)

		bust_coef = bust_coef * 0.9 + 0.1 * bust_coef_new
		boom_coef = boom_coef * 0.9 + 0.1 * boom_coef_new
		print(K_implied[5500:5900])

		beliefs = np.array([bust_coef,boom_coef])
		print(loss)
		print(beliefs)
		print(r2)
	return(beliefs)

def get_K_series(df_hh, agg_shocks):
	boom_years = pd.DataFrame(agg_shocks, columns = ["boom"]) # np.sum(boom_years)
	boom_years["year"] = boom_years.index
	K_implied = df_hh.groupby(["year"]).mean()["savings"].reset_index()
	K_implied = K_implied.merge(boom_years, left_on = "year", right_on = "year")
	boom_ix = K_implied[K_implied.boom].index
	bust_ix = K_implied[~K_implied.boom].index
	return(K_implied, boom_ix, bust_ix)

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
	
	policy_guess["policy_guess"] = policy_guess.index.get_level_values(3)
	df["policy_guess"] = policy_guess.index.get_level_values(3)
	
	# pol guess to something we can join to df
	
	diff = tol + 1
	
	while diff > tol:
		diff = 0
		
		policy_guess_mat, income_order, policy_guess_upd, df_pol = egm_policy(df, P, K_grid, policy_guess["policy_guess"], params)
		
		policy_guess_upd = policy_guess_upd.reset_index().set_index(["K_agg", "boom", "employed", "assets"])
		
		diff = max(np.max(np.abs(policy_guess_upd["policy_guess"] - policy_guess["policy_guess"])), diff)
		diff = max(policy_guess_upd["policy_guess"] - policy_guess["policy_guess"])
		policy_guess = policy_guess_upd.copy()

	return(policy_guess_mat, income_order, df_pol)

def egm_policy(df, P_df, K_grid, policy_guess, params):
	
	disc_factor = params["disc_factor"]
	risk_aver = params["risk_aver"]
	asset_grid_n = params["asset_grid"].shape[0]
	
	df = df.reset_index()
	df["mu_cons"] = ((1+df.rate) * df.assets + df.income - df.policy_guess)**(-params["risk_aver"])

	rate_today = df[["K_agg","boom", "rate"]].drop_duplicates().rename(columns = {"K_agg" : "K_agg_f", "boom" : "boom_f"}).set_index(["K_agg_f", "boom_f"])

	income_today = df[["K_agg","boom", "employed", "income"]].drop_duplicates().rename(columns = {"K_agg" : "K_agg_f", "boom" : "boom_f", "employed" : "employed_f"}).set_index(["K_agg_f", "boom_f", "employed_f"])
			
	assert df.mu_cons.isnull().sum() == 0, "There is negative consumption given policy."

	df_fut = df[["K_agg", "boom", "employed", "assets", "mu_cons", "rate"]]
	#df_fut = df_fut.set_index(["K_agg", "boom", "employed"])
	
	P_df["K_to"] = P_df["K_to"].astype("float64")

	P_df = P_df.reset_index().\
	rename(columns = {"boom" : "boom_f", "employed" : "employed_f", "K_from" : "K_agg_f"}).\
	rename(columns = {"boom_next" : "boom", "employed_next" : "employed", "K_to" : "K_agg"})

	df_fut = df_fut.merge(P_df, on = ["boom", "employed", "K_agg"])

	df_fut["exp_term"] = df_fut["prob"] * df_fut["mu_cons"] * (1 + df_fut["rate"])
	
	df_fut = df_fut[df_fut["exp_term"]!=0] # Exclude values made impossible by the K law of motion
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
		income_order[ss] = new_df.income.iloc[0]
		endog_x = np.asarray(new_df["endog_assets"])
		outcome = np.asarray(new_df.index.get_level_values(3))
		
		policy_mat[:,ss] = np.interp(x = exog_grid, xp = endog_x, fp = outcome)
		ss += 1
		
	new_policy_guess = pd.DataFrame(policy_mat, columns = ix_list)
	new_policy_guess["assets"] = exog_grid
	new_policy_guess = pd.melt(new_policy_guess, id_vars = "assets")
	new_policy_guess.index = pd.MultiIndex.from_tuples(new_policy_guess.variable)
	new_policy_guess = new_policy_guess.rename(columns = {"value" : "policy_guess"})
	new_policy_guess.index = new_policy_guess.index.set_names(["K_agg", "boom", "employed"])
	new_policy_guess = new_policy_guess[["policy_guess", "assets"]]

	df_pol = pd.DataFrame(policy_mat)
	df_pol.columns = income_order
	df_pol["assets"] = exog_grid

	return(policy_mat, income_order, new_policy_guess, df_pol)


shock_panel, agg_shocks = simulate_shocks(params["Nhh"], params["T"], params["P"])
		
df_inc = get_income_states(params["K_grid"], params["L_tilde"], params["alpha"])
df_rates = get_rates(params["K_grid"], params["L_tilde"], params["alpha"], params)
df = df_inc.merge(df_rates, 'left', left_on = ("K_agg", "boom"), right_on = ('K_agg', 'boom'))

start_belief = np.array([[0.095, 0.962], [0.085, 0.965]])
#start_belief = np.array([[0.2,0.9], [0.2,0.9]])

belief_star = find_eq(beliefs, shock_panel, agg_shocks, df, params)

P = get_transition_matrix(params["P"], params["K_grid"], beliefs)
hh_policy, income_order, df_pol = egm_dataframe(df, P, params["K_grid"], params["asset_grid"], params, tol = 1e-6)
hh_panel = hh_history_panel(shock_panel, agg_shocks, df, params["asset_grid"], hh_policy, income_order, params["L_tilde"], df_pol)
K_implied, boom_ix, bust_ix = get_K_series(hh_panel, agg_shocks)
boom_coef_new, r2 = update_beliefs(K_implied["savings"], boom_ix)
bust_coef_new, r2 = update_beliefs(K_implied["savings"], bust_ix)
# TODO LIST
# Steady state capital
# beliefs do not correspond with outcome



pd.DataFrame(hh_policy)



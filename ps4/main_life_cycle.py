# Main .py file for problem set 4, QMMI
import numpy as np
from numba import njit, prange
import pandas as pd
import sys
sys.path.append('../')
from life_cyc_calibrate import calibrate_life_cyc
from src.tauchenhussey import tauchenhussey, floden_w
from src.vfi_funcs import egm
from src.reward_funcs import calc_consumption
import plotly.express as px


# Part 1, Life cycle without bequests
# TODO: add point of the just borrowing constrained.
# TODO: endog_asset_states always give the same asset state conditional on year and income. 

# Calibration
params = calibrate_life_cyc(0.97)
exog_grid =np.repeat(np.expand_dims(params["action_states"], axis = 1), params["income_shock"].shape[0], axis = 1)

def lc_policies(params):
	
	V_guess = np.zeros((params["perm_inc_dim"] * params["trans_inc_dim"], params["action_size"])) # Not important. Should get rid off from egm.

	G_ret = params["determ_inc"].income.iloc[-1] * params["pensions_rr"]
	income_states = np.ones(params["income_shock"].shape[0]) * G_ret
	assets_endog = np.full((params["action_size"], params["income_shock"].shape[0]), 0.01/(1+params["rate"])) # Imposed in the PS

	min_age = params["determ_inc"].age.min()
	max_work_age = params["determ_inc"].age.max()
	max_age =  max_work_age + params["N_ret"]

	pol_mats = []
	endog_asset_states = []
	
	pol = np.zeros((params["action_size"], params["income_shock"].shape[0])) # Last policy is to save nothing

	pol_mats.append(pol)
	endog_asset_states.append(assets_endog)
	P = params["transition_matrix"]
	exog_grid =np.repeat(np.expand_dims(params["action_states"], axis = 1), params["income_shock"].shape[0], axis = 1)
	action_states = np.copy(exog_grid)
	for t in reversed(range(min_age, max_age-1)): # means it goes from 83 (inklusive) to 25 (inklusive), i.e. 59 values. 
		income_states_next = np.copy(income_states)
		
		if t > max_work_age:
			G = G_ret
			income_states = np.ones(params["income_shock"].shape[0]) * G
		if t <= max_work_age:
			G = params["determ_inc"][params["determ_inc"].age == t].income.iloc[0]
			income_states = params["income_shock"] * G
		
		disc_factor = params["disc_fact"] * params["surv_prob"][t-min_age+1]

		mu_cons_fut = ((1+params["rate"]) * action_states + income_states_next - pol)**(-params["risk_aver"])
		Ecf = np.matmul(mu_cons_fut, P.T)
		cons_today = (disc_factor * (1+params["rate"]) * Ecf)**(-1/params["risk_aver"])
		assets_endog = 1/(1+params["rate"]) * (cons_today + action_states - income_states)
		pol = np.empty(pol.shape)
		
		for s in range(income_states.shape[0]):
			pol[:,s] = np.interp(x = exog_grid[:,s], xp = assets_endog[:,s], fp = action_states[:,s], left = params["min_asset"], right = params["max_asset"])

		pol_mats.append(pol)
		endog_asset_states.append(assets_endog)
	
	pol_mats.reverse()
	endog_asset_states.reverse()

	pol_mats = np.stack(pol_mats)
	endog_asset_states = np.stack(endog_asset_states)

	return(pol_mats, endog_asset_states)

pol_mats, endog_asset_states = lc_policies(params)

# Part 2, simulate an economy
income_states_ix = np.arange(params["income_shock"].shape[0]) 

def create_shock_panel(params, min_age, max_age, income_states, income_states_ix):
	n_hh = params["n_hh"]
	years = max_age - min_age

	shock_matrix_ix = np.empty((years, n_hh), dtype = int) 
	shock_matrix_ix[0, :] = np.argmin(abs(income_states - 1)) # First income is 1
	trans_matrix = params["transition_matrix"]

	for hh in range(n_hh):
		for y in range(1, years):
			trans_prob = trans_matrix[shock_matrix_ix[y-1, hh]] # Index at previous state for conditional probabilities
			shock_matrix_ix[y, hh] = np.random.choice(income_states_ix, p = trans_prob)
			dies = np.random.uniform() > params["surv_prob"][y]
			if dies:
				shock_matrix_ix[range(y, years), hh] = -100
				break
	return(shock_matrix_ix)

def index_to_value_dict(index, values):
	state_dict = {}
	for ix, v in zip(index, values):
		state_dict[ix] = v
	return(state_dict)

def vec_translate(ix, state_dict):    
	return np.vectorize(state_dict.__getitem__)(ix)

min_age = params["determ_inc"].age.min()
max_work_age = params["determ_inc"].age.max()
max_age =  max_work_age + params["N_ret"]

shock_matrix_ix = create_shock_panel(params, min_age, max_age, params["income_shock"], income_states_ix) # TODO: make sure shape corresponds with inputs
# TODO: should be one more 
state_dict = index_to_value_dict(income_states_ix, params["income_shock"])
state_dict[-100] = np.nan # dead people
shock_matrix = vec_translate(shock_matrix_ix, state_dict)

def simulate_hh(n_hh, years, shock_matrix, income_states, endog_asset_states, pol_mats, min_age, max_age, max_work_age, params):
	def initiate_panel(years, n_hh):
		panel = np.empty((years, n_hh))
		panel[:] = np.nan
		return(panel)
	hh_panel_y = initiate_panel(years, n_hh)
	hh_panel_a = initiate_panel(years, n_hh)
	hh_panel_s = initiate_panel(years, n_hh)
	hh_panel_c = initiate_panel(years, n_hh)
	exog_grid =np.repeat(np.expand_dims(params["action_states"], axis = 1), params["income_shock"].shape[0], axis = 1)
	np.random.seed(1)  
	for hh in range(n_hh):
		
		initial_assets = min(np.exp(np.random.normal(-2.5, np.sqrt(4))), np.exp(-2.5 + 3 * 2))
		initial_income = 1 		
		
		income_state_ix = np.argmin(abs(income_states - initial_income)) # only initial income where initiated at 1
		asset_state_ix = np.argmin(abs(exog_grid[:, income_state_ix] - initial_assets)) 
		
		hh_panel_y[0, hh] = income_states[income_state_ix]
		hh_panel_a[0, hh] = exog_grid[asset_state_ix,income_state_ix]
		asset_next = pol_mats[0,asset_state_ix, income_state_ix]
		hh_panel_s[0, hh] = asset_next
		hh_panel_c[0, hh] = initial_income + initial_assets * (1+params["rate"]) - asset_next

		for y in range(1, years):
			if np.isnan(shock_matrix[y, hh]):
				break
			assets = asset_next
			if y + min_age > max_work_age:
				G = params["determ_inc"].income.iloc[-1] * params["pensions_rr"]
				income = G
			if y + min_age <= max_work_age:
				
				G = params["determ_inc"][params["determ_inc"].age == y + min_age].income.iloc[0]
				income = G * shock_matrix[y, hh]

			income_state_ix = np.argmin(abs(income_states * G - income))
			asset_state_ix = np.argmin(abs(exog_grid[:, income_state_ix] - assets))
			asset_next = pol_mats[y, asset_state_ix, income_state_ix]

			hh_panel_y[y, hh] = income
			hh_panel_a[y, hh] = assets
			hh_panel_s[y, hh] = asset_next
			hh_panel_c[y, hh] = income + assets * (1+params["rate"]) - asset_next
			
	return(hh_panel_y, hh_panel_a, hh_panel_s, hh_panel_c)

death_year = params["N_work"] + params["N_ret"] -1
inc_panel, ass_panel, save_panel, cons_panel = simulate_hh(5000, death_year, shock_matrix, params["income_shock"], endog_asset_states, pol_mats, min_age,  max_age, max_work_age, params)

# Part 3, plots
def plot_series(vector, label_dict):
	df = pd.DataFrame(np.nanmean(vector, axis = 1), columns = ["yvar"])
	df["age"] = df.index + min_age

	fig = px.line(df, x="age", y="yvar", template = 'plotly_white', labels = label_dict)
	return(fig)

# Life cycle profile for consunption, income asstes, share fo househoplds still alive. 
# CONSUMPTION
fig = plot_series(cons_panel, dict(age = "Age", yvar="# Goods"))
fig.update_layout(title = "Average consumption")
fig.write_image('figures/mean_cons.png')
# ASSETS
fig = plot_series(ass_panel, dict(age = "Age", yvar="# Assets"))
fig.update_layout(title = "Average asset holdings")
fig.write_image('figures/mean_assets.png')
# SAVINGS
fig = plot_series(save_panel, dict(age = "Age", yvar="# Assets"))
fig.update_layout(title = "Average savings")
fig.write_image('figures/mean_savings.png')
# INCOME
fig = plot_series(inc_panel, dict(age = "Age", yvar="# Goods"))
fig.update_layout(title = "Average income")
fig.write_image('figures/mean_income.png')
# DEAD PEOPLE
dead_panel = np.count_nonzero(~np.isnan(inc_panel), axis = 1) / inc_panel.shape[1]
dead_panel = np.expand_dims(dead_panel, axis = 1)
fig = plot_series(dead_panel, dict(age = "Age", yvar="Fraction"))
fig.update_layout(title = "Fraction Alive")
fig.write_image('figures/fraction_alive.png')


# Growth in average log consumption and average log income from j = 1 to the maximum lifecycle value
# What does he even mean?
# Cross-sectional variance of log consumption log income as households age
cons_var = np.nanvar(np.log(cons_panel), axis = 1)
inc_var = np.nanvar(np.log(inc_panel), axis = 1)
var_mat = np.array([cons_var, inc_var]).T
df = pd.DataFrame(var_mat, columns = ["Consumption", "Income"])
df["age"] = df.index + min_age
df = pd.melt(df, id_vars = "age", value_vars = ["Consumption", "Income"])
fig = px.line(df, x="age", y="value", color = "variable", 
	template = 'plotly_white', labels = dict(age = "Age", value="Variance"))
fig.update_layout(title = "Household Heterogeneity")
fig.write_image('figures/hh_heterogeneity.png')
fig.show()

# Part 4 insurance questions
# Construct objects below using all households in all time periods


# Vary rho, make sure this is a function
rhos = np.linspace(0.7, 0.995, 10)
insurance_rel = []
for rho in rhos:
	params = calibrate_life_cyc(rho)
	pol_mats, endog_asset_states = lc_policies(params)
	inc_panel, ass_panel, save_panel, cons_panel = simulate_hh(5000, death_year, shock_matrix, params["income_shock"], endog_asset_states, pol_mats, min_age,  max_age, max_work_age, params)
	diff_cons = np.diff(cons_panel[0:41], axis = 0) # ix_0 = ix_1 - ix_0
	shocks = shock_matrix[1:-19]
	cov_ = np.cov(diff_cons.flatten(), shocks.flatten())[0,1]
	var_ = np.var(shocks)
	phi = 1 - cov_/var_
	insurance_rel.append((rho, phi))

df = pd.DataFrame(insurance_rel, columns = ["Income shock persistence (ar1)", "phi"])
df["One minus phi"] = 1 - df["phi"]
fig = px.line(df, x="Income shock persistence (ar1)", y= "One minus phi", 
	template = 'plotly_white')
fig.show()
fig.write_image('figures/insurance_question.png')



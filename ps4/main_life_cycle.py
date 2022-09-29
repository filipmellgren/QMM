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

# Part 1, Life cycle without bequests
# TODO: add point of the just borrowing constrained.
# TODO: Implement death probability inside EGM
# TODO: compare income states

# Calibration
params = calibrate_life_cyc(0.97)

def lc_policies(params):
	
	V_guess = np.zeros((params["perm_inc_dim"] * params["trans_inc_dim"], params["action_size"])) # Not important. Should get rid off from egm.

	G_ret = params["determ_inc"].income.iloc[-1] * params["pensions_rr"]
	income_states = params["income_shock"] * G_ret #np.repeat(G_ret, params["income_shock"].shape[0]) # No shocks to the pensioners :)
	assets_endog = np.full((params["action_size"], params["income_shock"].shape[0]), 0.01/(1+params["rate"])) # Imposed in the PS

	min_age = params["determ_inc"].age.min()
	max_work_age = params["determ_inc"].age.max()
	max_age =  max_work_age + params["N_ret"]

	pol_mats = []
	endog_asset_states = []
	
	pol = np.zeros((params["action_size"], params["income_shock"].shape[0]))

	pol_mats.append(pol)
	endog_asset_states.append(assets_endog)

	for t in reversed(range(min_age, max_age-1)):
		action_states = np.copy(assets_endog)
		income_states_next = np.copy(income_states)
		
		if t > max_work_age:
			G = G_ret
		if t <= max_work_age:
			G = params["determ_inc"][params["determ_inc"].age == t].income.iloc[0]
		
		income_states = params["income_shock"] * G # Strictly, retirees never experience shocks, they can still have policies for it.
		
		disc_factor = params["disc_fact"] * params["surv_prob"][t-min_age] # TODO: do I use this correctly, or should it be shifted one period?
		
		V, pol, assets_endog = egm(V_guess.T, params["transition_matrix"], action_states, 
			income_states, income_states_next, params["rate"], disc_factor, params, tol = 1e-6)
		
		pol_mats.append(pol)
		endog_asset_states.append(assets_endog)
	pol_mats = np.stack(pol_mats)
	endog_asset_states = np.stack(endog_asset_states)

	return(pol_mats, endog_asset_states)

pol_mats, endog_asset_states = lc_policies(params)

# TODO: Divide thorugh by permamenent income. Has this been done already inside EGM??

# Part 2, simulate an economy
income_states_ix = np.arange(params["income_shock"].shape[0]) 

def create_shock_panel(params, min_age, max_age, income_states, income_states_ix):
	# Only persistent shocks
	n_hh = params["n_hh"]
	years = max_age - min_age - 1

	shock_matrix_ix = np.empty((years, n_hh), dtype = int) 
	shock_matrix_ix[0, :] = np.argmin(abs(income_states - 1)) # First income is 1
	trans_matrix = params["transition_matrix"]

	for y in range(1, years):
		for hh in range(n_hh):
			trans_prob = trans_matrix[shock_matrix_ix[y-1, hh]] # Index at previous state for conditional probabilities
			shock_matrix_ix[y, hh] = np.random.choice(income_states_ix, p = trans_prob)
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

shock_matrix_ix = create_shock_panel(params, min_age, max_age, params["income_shock"], income_states_ix)
state_dict = index_to_value_dict(income_states_ix, params["income_shock"])
shock_matrix = vec_translate(shock_matrix_ix, state_dict)

# for hh in range(n_hh):
#@njit()
def simulate_hh(n_hh, years, shock_matrix, income_states, endog_asset_states, pol_mats, min_age, max_age, max_work_age, params):
	hh_panel_y = np.zeros((years, n_hh))
	hh_panel_a = np.zeros((years, n_hh))
	hh_panel_s = np.zeros((years, n_hh))
	hh_panel_c = np.zeros((years, n_hh))

	for hh in range(n_hh):
		initial_assets = min(np.exp(np.random.normal(-2.5, np.sqrt(4))), np.exp(-2.5 + 3 * 2))
		initial_income = 1 		

		income_state_ix = np.argmin(abs(income_states - initial_income))
		asset_state_ix = np.argmin(abs(endog_asset_states[0,:,income_state_ix] - initial_assets)) 
		
		hh_panel_y[0, hh] = income_states[income_state_ix]
		hh_panel_a[0, hh] = initial_assets
		asset_next = pol_mats[0, asset_state_ix, income_state_ix]
		hh_panel_s[0, hh] = asset_next
		hh_panel_c[0, hh] = initial_income + initial_assets * (1+params["rate"]) - asset_next

		for y in range(1, years):
			
			if y + min_age > max_work_age:
				G = params["determ_inc"].income.iloc[-1] * params["pensions_rr"]
				income = G
			if y + min_age <= max_work_age:
				G = params["determ_inc"][params["determ_inc"].age == y + min_age].income.iloc[0]
				income = G * shock_matrix[y, hh]

			assets = asset_next
			asset_next = pol_mats[y, asset_state_ix, income_state_ix]

			hh_panel_y[y, hh] = income
			hh_panel_a[y, hh] = assets
			hh_panel_s[y, hh] = asset_next
			hh_panel_c[y, hh] = income + assets * (1+params["rate"]) - asset_next

			income_state_ix = np.argmin(abs(income_states - income))
			asset_state_next = pol_mats[y, income_state_ix, asset_state_ix]
			asset_state_ix = np.argmin(abs( - asset_state_next))
			
	return(hh_panel_y, hh_panel_a, hh_panel_s, hh_panel_c)

death_year = params["N_work"] + params["N_ret"] - 1
inc_panel, ass_panel, save_panel, cons_panel = simulate_hh(5000, death_year, shock_matrix, params["income_shock"], endog_asset_states, pol_mats, min_age,  max_age, max_work_age, params)


# Part 3, plots
import plotly.express as px

def plot_series(vector):
	df = pd.DataFrame(np.mean(vector, axis = 1), columns = ["mean_income"])
	df["age"] = df.index + min_age

	fig = px.line(df, x="age", y="mean_income", 
		title='Mean income over time', template = 'plotly_white',
		labels=dict(age = "Age", mean_income="Average income"))
	return(fig)

# Life cycle profile for consunption, income asstes, share fo househoplds still alive. 

fig = plot_series(inc_panel)
fig.show()
fig.write_image('figures/mean_income.png')



# Growth in average log consumption and average log income from j = 1 to the maximum lifecycle value

# Cross-sectional variance of log consumption log income as households age

# Part 4 insurance questions

# Construct objects below using all households in all time periods
phi = 1 - np.cov(D_cons, pis)/np.cov(pis, pis)

# Vary rho, make sure this is a function


## TMP
age_index = 35 - 25 - 1
asset_index = 550 - 1
det_inc = params["determ_inc"]

income_35 = np.asarray(det_inc[det_inc.age == 35].income) * params["income_shock"]
endog_asset_states[age_index, asset_index+1] * (1+ params["rate"]) + income_35 - pol_mats[age_index, asset_index,:]



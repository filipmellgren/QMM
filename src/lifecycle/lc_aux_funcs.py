import numpy as np
from numba import njit
import scipy as sp
import ipdb
def lc_policies(params, income_states_matrix, transition_matrix, phi1):
	'''
	TODO: implement possibility for bequests in one period and update transition matrix basex on this.
	An idea is to do the trick of extending the income states by factor 20, but keeping values the same in most periods. 
	'''
	terminal_income = params["terminal_income_states"]
	
	min_age = params["min_age"]
	max_work_age = params["max_work_age"]
	max_age =  params["max_age"]
	G_ret = params["determ_inc"].income.iloc[-1] * params["pensions_rr"]

	pol_mats = []
	terminal_policy = params["terminal_policy"]
	pol_mats.append(terminal_policy)

	pol = terminal_policy
	income_states = terminal_income
	years = params["N_work"] + params["N_ret"]
	for t in reversed(range(years-1)):

		disc_factor = params["disc_fact"] * params["surv_prob"][t+1]
		income_states_next = np.copy(income_states)
		
		income_states = income_states_matrix[t]
		
		mu_cons_fut = ((1+params["rate"]) * params["exog_grid"] + income_states_next - pol)**(-params["risk_aver"])
		
		Bs = (1-params["estate_tax"]) * params["exog_grid"]

		mu_beq_fut = (1 - params["estate_tax"]) * (1 - params["risk_aver"]) * phi1 *(1 + Bs/params["phi2"])**(-params["risk_aver"])/params["phi2"]

		mu_fut = mu_cons_fut + mu_beq_fut


		pol = EGM(mu_fut, disc_factor, params["exog_grid"], income_states, income_states_next, pol,transition_matrix, params)
		pol_mats.append(pol)
		
	pol_mats.reverse()
	pol_mats = np.stack(pol_mats)

	return(pol_mats)


def create_income_states(mltp_shock, det_income, params):
	max_work_age = params["max_work_age"]
	retirement = params["determ_inc"].income.iloc[-1] * params["pensions_rr"]
	income_states_mat = []
	years = params["N_work"] + params["N_ret"]

	for t in range(years):
		if t >= max_work_age - params["min_age"] + 1:
			G = retirement
			income_states = G * np.ones(mltp_shock.shape[0])

		if t < max_work_age - params["min_age"] + 1:
			
			G = params["determ_inc"][params["determ_inc"].age == t + params["min_age"] ].income.iloc[0]
			income_states = mltp_shock
			income_states = income_states * G

		income_states_mat.append(income_states)
	income_states_mat = np.stack(income_states_mat)

	return(income_states_mat)

def create_hh_bequests(bequest_shock_probs, guess, params):
	hh_bequests = []
	for hh in range(params["n_hh"]):
		y = params["N_work"] # When everyone is bequested in this economy
		bequest_size = np.argmin(np.abs(np.random.uniform() - bequest_shock_probs))	
		
		bequest = params["bequest_grid"][bequest_size] 
		bequest = min(bequest, guess[1] + 3 * np.sqrt(guess[2])) * (1 - params["estate_tax"]) 
		hh_bequests.append(bequest)
	hh_bequests = np.stack(hh_bequests)
	return(hh_bequests)

def EGM(mu_cons_fut, disc_factor, action_states, income_states, income_states_next, pol, P, params):
	'''
	P: transition matrix. Rows indicate from, columns indicate to. Row stochastic matrix. 
	TODO: should allow for different utility functions
	'''
	
	Ecf = np.matmul(mu_cons_fut, P.T)
	cons_today = (disc_factor * (1+params["rate"]) * Ecf)**(-1/params["risk_aver"])
	
	assets_endog = 1/(1+params["rate"]) * (cons_today + action_states - income_states)
	pol = np.empty(pol.shape)
	for s in range(income_states.shape[0]):
		pol[:,s] = np.interp(x = action_states[:,s], xp = assets_endog[:,s], fp = action_states[:,s], left = params["min_asset"], right = params["max_asset"])
	return(pol)



def create_shock_panel(params, transition_matrix, min_age, max_age):
	''' Simulates based on indices, then convert to values
	'''
	income_states_ix = np.arange(params["income_shock"].shape[0])
	n_hh = params["n_hh"]
	years = max_age - min_age

	shock_matrix_ix = np.empty((years, n_hh), dtype = int) 
	shock_matrix_ix[0, :] = np.argmin(abs(params["income_shock"] - 1)) # First income is 1

	for hh in range(n_hh):
		for y in range(1, years):
			trans_prob = transition_matrix[shock_matrix_ix[y-1, hh]] # Index at previous state for conditional probabilities
			shock_matrix_ix[y, hh] = np.random.choice(income_states_ix, p = trans_prob)
			dies = np.random.uniform() > params["surv_prob"][y]
			if dies:
				shock_matrix_ix[range(y, years), hh] = -100
				break
	state_dict = index_to_value_dict(income_states_ix, params["income_shock"])
	state_dict[-100] = np.nan # dead people
	shock_matrix = vec_translate(shock_matrix_ix, state_dict)
		
	return(shock_matrix)

def index_to_value_dict(index, values):
	state_dict = {}
	for ix, v in zip(index, values):
		state_dict[ix] = v
	return(state_dict)

def vec_translate(ix, state_dict):    
	return np.vectorize(state_dict.__getitem__)(ix)


def get_bequest_transition_matrix(guess, params):
	bequest_grid = params["bequest_grid"]
	bequest_grid_mid_points = bequest_grid + np.mean(np.diff(bequest_grid))/2 # Add half step size
	bequest_shock_cdf= sp.stats.lognorm.cdf(bequest_grid_mid_points, s = guess[2], loc = guess[1])
	bequest_shock_probs = np.diff(np.append(0, bequest_shock_cdf)) # TODO: almost 1 what to do?
	bequest_shock_probs = bequest_shock_probs / np.sum(bequest_shock_probs) # Scale to 1
	bequest_trans_mat = np.tile(bequest_shock_probs, (bequest_shock_probs.shape[0], 1))
	transition_matrix = np.kron(params["transition_matrix"], bequest_trans_mat)
	return(transition_matrix, bequest_shock_probs)

def simulate_hh(n_hh, income_states, shock_matrix, pol_mats, hh_bequests, params):
	'''
	TODO: make easier to read by creating a multidimensional array
	# TODO: referencing the income states like I do with index seems weak. 
	'''
	years = params["N_work"] + params["N_ret"] -1
	def initiate_panel(years, n_hh):
		panel = np.empty((years, n_hh))
		panel[:] = np.nan
		return(panel)
	min_age = params["min_age"]
	max_work_age = params["max_work_age"]

	hh_panel_y = initiate_panel(years, n_hh)
	hh_panel_a = initiate_panel(years, n_hh)
	hh_panel_s = initiate_panel(years, n_hh)
	hh_panel_c = initiate_panel(years, n_hh)
	
	exog_grid = params["exog_grid"] 

	np.random.seed(1)  
	for hh in range(n_hh):
		
		initial_assets = min(np.exp(np.random.normal(-2.5, np.sqrt(4))), np.exp(-2.5 + 3 * 2))
		initial_income = 1 		
		
		income_state_ix = np.argmin(abs(income_states[0] - initial_income)) # only initial income where initiated at 1
		asset_state_ix = np.argmin(abs(exog_grid[:, income_state_ix] - initial_assets)) 
		
		hh_panel_y[0, hh] = income_states[0][income_state_ix]
		hh_panel_a[0, hh] = exog_grid[asset_state_ix,income_state_ix]
		asset_next = pol_mats[0,asset_state_ix, income_state_ix]
		hh_panel_s[0, hh] = asset_next
		hh_panel_c[0, hh] = initial_income + initial_assets * (1+params["rate"]) - asset_next

		income_vector = np.asarray(params["determ_inc"].income)
		bequests = hh_bequests[hh]

		hh_panel_y, hh_panel_a, hh_panel_s, hh_panel_c = sim_years(hh, asset_next, years, income_vector, income_states, params["pensions_rr"], shock_matrix, pol_mats, exog_grid, min_age, max_work_age, params["rate"], hh_panel_y, hh_panel_a, hh_panel_s, hh_panel_c, bequests)
			 
	return(hh_panel_y, hh_panel_a, hh_panel_s, hh_panel_c)

@njit
def sim_years(hh, asset_next, years, income_vector, income_states, pension_factor, shock_matrix, pol_mats, exog_grid, min_age, max_work_age, rate, hh_panel_y, hh_panel_a, hh_panel_s, hh_panel_c, bequests):
	for y in range(1, years):
		assets = asset_next
		if np.isnan(shock_matrix[y, hh]):
			break # dead TODO: report back amount of assets perhaps
		if y + min_age > max_work_age + 1:
			G = income_vector[-1] * pension_factor
			income = G
		if y + min_age == max_work_age + 1:
			G = income_vector[-1] * pension_factor
			income = G #+ bequests
			assets += bequests
		if y + min_age < max_work_age + 1:
			G = income_vector[y]
			income = G * shock_matrix[y, hh] 

		income_state_ix = np.argmin(np.abs(income_states[y] - income)) 
		asset_state_ix = np.argmin(np.abs(exog_grid[:, income_state_ix] - assets))

		asset_next = pol_mats[y, asset_state_ix, income_state_ix]

		hh_panel_y[y, hh] = income
		hh_panel_a[y, hh] = assets
		hh_panel_s[y, hh] = asset_next
		hh_panel_c[y, hh] = income + assets * (1+rate) - asset_next
	return(hh_panel_y, hh_panel_a, hh_panel_s, hh_panel_c)


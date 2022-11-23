
import numpy as np
from numba import jit, prange
import numba as nb
from market_class import assets_from_state, is_employed
import ipdb

def solve_hh(P, rate, wage, tax, labor, mu, risk_aver, disc_factor, delta, state_grid, asset_states):
	''' Solve the HH problem
	Returns a policy vector. One savings value per possible state. 
	Comes very close to quantecon's  policy iteration values. Sometimes the policy is slightly too high relative to quantecon (never too low). Most often the same. 
	# Iterate on values. Return index. 
	'''
	policy_guess = np.asarray([asset_states, asset_states])
	tol = 1e-13 # 1e-3 Smaller than minimum distance on grid
	
	diff = 100
	iteration = 0
	while diff > tol:
		iteration += 1
		policy_guess_upd = egm_update(policy_guess, P, rate, rate, wage, wage, tax, labor, mu, risk_aver, disc_factor, delta, state_grid, asset_states)

		if iteration % 25 == 0:
			diff = np.max(np.abs(policy_guess_upd - policy_guess))
		
		policy_guess = policy_guess_upd.copy()
	
	policy = policy_guess.flatten()
	return(policy)

def value_array_to_index(value_array, grid):
	array_ix = []
	for val in value_array:
		nearest_ix = np.argmin(np.abs(val - grid))
		array_ix.append(nearest_ix)
	return(array_ix)

def policy_to_grid(policy, asset_states):
	''' Put endopegnous policy on grid
	policy is a flat array
	Allocates points in between grid points to both grid points by putting fraction 
	alpha above, and 1 - alpha below. 
	alpha = (a_up - pol) /(a_up - a_down)
	'''
	

	policy_ix_up = []
	alpha_list = []
	nearest_ix_list = []
	for pol in policy:
		nearest_ix = np.argmin(np.abs(pol - asset_states)) # takes closest value
		nearest_ix_list.append(nearest_ix)
		rounded_down = pol >= asset_states[nearest_ix]
		if rounded_down:
			closest_above = nearest_ix + 1
			closest_below = nearest_ix
		if not rounded_down:
			closest_above = nearest_ix
			closest_below = nearest_ix - 1
		policy_ix_up.append(closest_above)
		if closest_above > 0:
			try:
				alpha =  (asset_states[closest_above] - pol)/(asset_states[closest_above] - asset_states[closest_below])			
				alpha_list.append(alpha)
			except IndexError:
				alpha_list.append(0.0)
			continue
		alpha_list.append(0.0)
	policy_ix_up = np.asarray(policy_ix_up)
	policy_ix_up[policy_ix_up >= len(asset_states)] = len(asset_states) - 1
	
	return(policy_ix_up, alpha_list)

@jit(nopython=True)
def mu_cons(consumption, risk_aver):
	return(consumption**(-risk_aver))

@jit(nopython=True, parallel = False)
def egm_update(policy_guess, P, rate, rate_fut, wage, wage_fut, tax, labor, mu, risk_aver, disc_factor, delta, state_grid, asset_states):
	''' Use EGM to solve the HH problem.

	policy_guess should be values, not indices, in matrix format, not vector

	'''
	# From Utility to Marginal Utility with CRRA preferences:
	#mu_cons = ((1 - risk_aver) * R - 1)**(-risk_aver / (1 - risk_aver))
	#mu_cons = (savings + assets*(1+r-delta) + income)**(-risk_aver)
	
	policy_mat = np.empty(policy_guess.shape)
	grid_density = asset_states[1] - asset_states[0]

	#for state in range(len(state_grid)):
	for employed in [0,1]:
		
		income = wage * ((1-tax) * labor * employed + mu * (1 - employed))
		endog_assets = []

		for action in range(len(asset_states)):
			E_term = 0
			savings = asset_states[action]
			for state_next in range(len(state_grid)):
				# Test whether state_next is reached by action taken before proceeding:
				assets_fut = assets_from_state(state_next, asset_states)
				if np.abs(assets_fut - savings) > grid_density:
					continue
				assets_fut_ix = np.argmin(np.abs(assets_fut - asset_states))
				employed_fut = is_employed(state_next, state_grid)

				savings_fut = policy_guess[int(employed_fut), assets_fut_ix]
				income_fut = wage_fut * ((1-tax) * labor * employed_fut + mu * (1 - employed_fut))
				consumption_fut = income_fut + assets_fut * (1+ rate_fut - delta)- savings_fut 

				mu_cons_fut = mu_cons(consumption_fut, risk_aver)
				prob = P[int(employed), int(employed_fut)]
				E_term += prob * mu_cons_fut * (1 + rate_fut - delta) 
			
			consumption_today = (disc_factor * E_term)**(-1/risk_aver) # Rate already in expectation
			# Mapping from actions to assets_endog, given state:
			endog_assets.append((1/(1+rate-delta)) * (consumption_today + savings - income))

		endog_assets = np.asarray(endog_assets)
		policy_mat[employed,:] = np.interp(x = asset_states, xp = endog_assets, fp = asset_states)

	return(policy_mat)








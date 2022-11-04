# Iterate to solve the firm's value functions
import numpy as np
from firm_choices import intermediate_good_price, firm_choices
def V0_of_xi(xi, price_guess, adj_val, omega, q, inventory_grid, V1_guess):
	''' Eq 11
	TODO: Do we need this?
	Maybe deprecate, maybe use later
	'''
	net_adj_val = adj_val - price_guess * omega * xi
	net_cont_val = -price_guess * q * inventory_grid + V1_guess
	V0 = price_guess * q * inventory_grid + np.maximum(net_adj_val, net_cont_val) # Eq. 11
	return(V0)

def firm_vf_upd(V1, price_guess, adj_val, q, inventory_grid, omega, params):
	''' Update the firm's expectation of the value function V0
	'''
	
	xi_min = params["xi_min"]
	xi_max = params["xi_max"]
	#ipdb.set_trace()
	xi_tilde = -(-price_guess * q * inventory_grid + V1 - adj_val)/(price_guess * omega) # Eq 12
	xi_T = np.minimum(np.maximum(xi_min, xi_tilde), xi_max)
	adj_share = (xi_T-xi_min)/(xi_max - xi_min) # Uniform assumption "H"
	E_xi = (xi_T**2)/(2*xi_max) # TODO: confirm this is expectation of xi between xi_min and xi_T

	EV0_upd = adj_share * (price_guess * q * inventory_grid + adj_val) - price_guess * omega * E_xi + (1-adj_share) * V1 # Eq 13

	return(EV0_upd, adj_share)

def iterate_firm_vf(price_guess, inventory_grid, EV0_guess, tol, params):
	''' Iterate on firm's value function to find true value
	'''
	q = intermediate_good_price(price_guess, params)
	omega = params["eta"]/price_guess
	diff = 100.0
	
	while diff > tol:
		inventory_star, adj_val, m_star, V1_upd = firm_choices(inventory_grid, price_guess, EV0_guess, params)
		try:
			V1_diff = np.max(np.abs(V1_guess - V1_upd))
		except NameError as e:
			V1_diff = 100.0 # For first iteration
		V1_guess = np.copy(V1_upd)
		EV0_upd, adj_share = firm_vf_upd(V1_guess, price_guess, adj_val, q, inventory_grid, omega, params)
		EV0_diff = np.max(np.abs(EV0_upd - EV0_guess))
		EV0_guess = np.copy(EV0_upd)

		diff = np.maximum(V1_diff, EV0_diff)
	return(EV0_guess, V1_guess, inventory_star, m_star, adj_share, adj_val)

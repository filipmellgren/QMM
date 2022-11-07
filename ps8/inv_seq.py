# Inventory sequences
from firm_vf import iterate_firm_vf
from firm_choices import intermediate_good_price

import numpy as np
from scipy.interpolate import UnivariateSpline

def calc_adj_share(s_, V1_spline, price_guess, q, adj_val, omega, xi_min, xi_max):
	V1_s = V1_spline(s_)
	xi_tilde = -(V1_s - price_guess * q * s_ - adj_val)/(price_guess * omega)
	xi_T = np.minimum(np.maximum(xi_min, xi_tilde), xi_max)
	adj_share = (xi_T**2)/(2 * xi_max)
	return(adj_share)

def find_inventory_seq(price_guess, params):
	q = intermediate_good_price(price_guess, params)
	omega = params["eta"]/price_guess
	xi_min = params["xi_min"]
	xi_max = params["xi_max"]
	inventory_grid = params["inventory_grid"]

	EV0_guess = price_guess**(1/(1-params["theta_m"])) * (1 - params["theta_n"]) * (params["theta_n"]/params["eta"])**(params["theta_n"]/(1-params["theta_n"])) * inventory_grid**(params["theta_m"]/(1-params["theta_n"]))

	EV0, V1, inventory_star, m_star, adj_share, adj_val = iterate_firm_vf(price_guess, inventory_grid, EV0_guess, 1e-6, params)
	#EV0_spline = UnivariateSpline(inventory_grid, EV0) 
	V1_spline = UnivariateSpline(inventory_grid, V1)
	
	s_ = inventory_star
	inv_seq = []
	adj_share_seq = []
	m_seq = []
	eps0 = 1e-8
	all_firms_updated = False
	while not all_firms_updated:
		inv_seq.append(s_)
		adj_share_seq.append(calc_adj_share(s_, V1_spline, price_guess, q, adj_val, omega, xi_min, xi_max))
		if s_ < eps0:
			m_seq.append(s_)
			s_ = 0
			all_firms_updated = np.sum(adj_share_seq) >= 1
			continue

		m_ = np.interp(s_, inventory_grid, m_star) # TODO: update to smoother spline 
		m_seq.append(m_)
		s_ = s_ - m_
		all_firms_updated = np.sum(adj_share_seq) >= 1

	return(inv_seq, m_seq, adj_share_seq)

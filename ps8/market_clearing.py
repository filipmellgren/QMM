from final_good_distribution import find_ergodic_distribution, find_distribution_loop
import numpy as np
from firm_choices import final_good_production
import ipdb

def market_clearing(price_guess, params):
	sigma = params["sigma"]
	theta_n = params["theta_n"]
	theta_m = params["theta_m"]
	eta = params["eta"]
	beta = params["beta"]
	zbar = params["zbar"]
	alpha = params["alpha"]
	delta = params["delta"]
	wage = eta / price_guess
	params["wage"] = wage

	distr, m_vec, s_vec = find_distribution_loop(price_guess, params)
	
	intermediate_good_demand = m_vec @ distr
	X = intermediate_good_demand

	# From w = MPL, get (K/L)**alpha ratio
	
	KLalpha_ratio = beta * alpha /(1-alpha) * wage /(1- beta * (1-delta))
	KLalpha_ratio = KLalpha_ratio ** (1/alpha)
	# Solve for L
	labor_ig = X / (zbar * KLalpha_ratio)
	K = labor_ig * (wage /(zbar * (1-alpha)))**(1/alpha)
	
	n_vec = (theta_n/eta * price_guess * m_vec**theta_m)**(1/(1-theta_n))
	G = final_good_production(m_vec, n_vec, params)

	consumption = np.sum((G - sigma * (s_vec - m_vec))*distr) - delta * K 
	
	marginal_utility = 1/consumption
	price_preferences = marginal_utility
	diff = price_guess - price_preferences
	
	return(diff, distr, m_vec, s_vec)


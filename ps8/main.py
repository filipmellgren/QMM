# Firms and inventories
# By filip.mellgren@su.se
import numpy as np
from numpy import linalg as LA
from firm_choices import firm_choices, intermediate_good_price
from final_good_distribution import find_ergodic_distribution
from firm_choices import final_good_production
from scipy import optimize
import scipy as sp
from inv_seq import find_inventory_seq
import ipdb


params = {
	"beta" : 0.984,
	"eta" : 2.128,
	"alpha" : 0.3739,
	"theta_m" : 0.4991,
	"theta_n" : 0.3275,
	"delta" : 0.0173,
	"eta_bar" : 0.2198,
	"zbar" : 1.0032,
	"sigma" : 0.012,
	"xi_min" : 0,
	"xi_max" : 0.2198
	}


psi0 = 10
psi1 = 25
psi0_grid = np.linspace(
	start = np.log(0.1042/psi1)/np.log(psi0),
	stop =  np.log(2.5) / np.log(psi0),
	num = 24)
inventory_grid = np.append(np.array([0]), psi0**psi0_grid)
params["inventory_grid"] = inventory_grid

p_guess = 3.25
price_guess = 3.25



def market_clearing(price_guess, params):
	sigma = params["sigma"]
	theta_n = params["theta_n"]
	theta_m = params["theta_m"]
	eta = params["eta"]
	zbar = params["zbar"]
	alpha = params["alpha"]
	delta = params["delta"]
	wage = eta / price_guess

	distr, m_vec, s_grid, s_vec = find_ergodic_distribution(price_guess, params)
	distr = np.squeeze(distr)

	intermediate_good_demand = m_vec @ distr
	X = intermediate_good_demand

	# From w = MPL, get (K/L)**alpha ratio
	KLalpha_ratio = wage / (zbar * (1 - alpha))
	# Solve for L
	labor_ig = X / (zbar * KLalpha_ratio)
	K = labor_ig * (wage /(zbar * (1-alpha)))**(1/alpha)

	n_vec = (theta_n/eta * price_guess * m_vec**theta_m)**(1/(1-theta_n))
	G = final_good_production(m_vec, n_vec, params)

	consumption = np.sum((G - sigma * (s_vec - m_vec))*distr) - delta * K

	marginal_utility = 1/consumption
	price_preferences = marginal_utility
	diff = price_guess - price_preferences
	print(diff)
	return(diff)

market_clearing(3.2+0.001, params)
market_clearing(3.3-0.001, params)



market_clearing(3.25, params)
market_clearing(3.3-0.001, params)
market_clearing(4, params)

price_star = sp.optimize.bisect(market_clearing, 3.2, 3.3, args = params, xtol = 1e-8)


np.around(P,4)






#### START



len(m_seq)
LA.eig(adj_share_seq)




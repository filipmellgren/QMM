# Firms and inventories
# By filip.mellgren@su.se
import numpy as np
from numpy import linalg as LA
from firm_choices import firm_choices, intermediate_good_price
from final_good_distribution import find_ergodic_distribution

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

distr, m_vec = find_ergodic_distribution(price_guess, params)


def market_clearing(price_guess, distr, params):
	intermediate_good_demand = m_vec @ distr
	capital = 
	labor
	# TODO: we should sum up to J_max, but I lump them together already. Is that ok?
	consumption = np.sum(G - sigma * (s_vec - m_vec))


	marginal_utility = 1/consumption
	price_preferences = marginal_utility
	diff = price_guess - price_preferences

	return(diff)



sp.optimize.bisect()


np.around(P,4)






#### START



len(m_seq)
LA.eig(adj_share_seq)




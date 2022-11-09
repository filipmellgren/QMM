# Firms and inventories
# By filip.mellgren@su.se
import numpy as np
from market_clearing import market_clearing
from scipy import optimize
import scipy as sp
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

price_guess = 3.22

market_clearing(price_guess, params)




price_guess = 3.291
market_clearing(price_guess, params)

price_guess = 3.22
market_clearing(price_guess, params)

price_star = sp.optimize.bisect(market_clearing, 3.22, 3.26, args = params, xtol = 1e-8)

'''
For the final good distribution, do you guys consider firms at s* at all? He says to only start at s* - m(s*) and then look at the firms not adjusting - but what about the firms that do adjust back to s*? They have to be in our distribution right?

The not adjusting firms make the mass at the related point and the sum of the adjusting firms gives the mass at s*

No firm at s* adjusts so you start from one level below
'''
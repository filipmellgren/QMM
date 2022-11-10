# Firms and inventories
# By filip.mellgren@su.se
import numpy as np
from market_clearing import market_clearing
from scipy import optimize
import scipy as sp
import pandas as pd

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

def simulate_inventory_mgmt(path, params):
	price_star = sp.optimize.bisect(lambda x: market_clearing(x, params)[0], 3.1, 3.4, xtol = 1e-8)
	diff, distr, m_vec, s_vec = market_clearing(price_star, params)
	price_star = np.around(price_star,4)
	table = pd.DataFrame(np.array([s_vec, distr, m_vec])).round(5)
	table.index = ["s", "Mass", "m"]
	table.to_csv(path + "_table.csv")
	np.savetxt(path + "_price.csv", np.expand_dims(price_star, axis = 0))
	return

simulate_inventory_mgmt("figures/low_ximax", params)
params["xi_max"] = 0.333
simulate_inventory_mgmt("figures/high_ximax", params)
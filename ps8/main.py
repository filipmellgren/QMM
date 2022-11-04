# Firms and inventories
# By filip.mellgren@su.se
import numpy as np
from firm_choices import firm_choices, intermediate_good_price
from firm_vf import iterate_firm_vf
import ipdb
from scipy.interpolate import UnivariateSpline

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

p_guess = 3.25
price_guess = 3.25

EV0_guess = p_guess**(1/(1-params["theta_m"])) * (1 - params["theta_n"]) * (params["theta_n"]/params["eta"])**(params["theta_n"]/(1-params["theta_n"])) * inventory_grid**(params["theta_m"]/(1-params["theta_n"]))

#### START

EV0, V1, inventory_star, m_star, adj_share, adj_val = iterate_firm_vf(price_guess, inventory_grid, EV0_guess, 1e-6, params)

q = intermediate_good_price(price_guess, params)
omega = params["eta"]/price_guess
xi_min = params["xi_min"]
xi_max = params["xi_max"]

EV0_spline = UnivariateSpline(inventory_grid, EV0, k =3) 
V1_spline = UnivariateSpline(inventory_grid, V1, k=3)

xi_tilde_star = -(V1_spline(inventory_star) - price_guess * q * inventory_star - adj_val)/(price_guess * omega)
xi_T_star = np.minimum(np.maximum(xi_min, xi_tilde_star), xi_max)








def find_inventory_seq(s_star, adj_share):
	eps0 = 0.01
	EV0, V1, inventory_star, m_star, adj_share = iterate_firm_vf(price_guess, inventory_grid, EV0_guess, 1e-6, params)
	EV0_spline = UnivariateSpline(inventory_grid, EV0) 
	V1_spline = UnivariateSpline(inventory_grid, V1)
	# TODO: should we loop here and is J_max time
	for time in range(J_max):
		s_new = inventory_grid - m_star
		s_new = s_new[s_new < eps0]

		# TODO: check if all firms have updated
		all_firms_updated = False
		if all_firms_updated:
			continue
			pass
			pass
		pass

	pass





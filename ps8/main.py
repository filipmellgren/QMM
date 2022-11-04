# Firms and inventories
# By filip.mellgren@su.se
import numpy as np
from firm_choices import firm_choices

params = {
	"beta" : 0.984,
	"eta" : 2.128,
	"alpha" : 0.3739,
	"theta_m" : 0.4991,
	"theta_n" : 0.3275,
	"delta" : 0.0173,
	"eta_bar" : 0.2198,
	"zbar" : 1.0032,
	"sigma" : 0.012
	}

psi0 = 10
psi1 = 25
psi0_grid = np.linspace(
	start = np.log(0.1042/psi1)/np.log(psi0),
	stop =  np.log(2.5) / np.log(psi0),
	num = 24)
inventory_grid = np.append(np.array([0]), psi0**psi0_grid)

p_guess = 3.25

EV0_guess = p_guess**(1/(1-params["theta_m"])) * (1 - params["theta_n"]) * (params["theta_n"]/params["eta"])**(params["theta_n"]/(1-params["theta_n"])) * inventory_grid**(params["theta_m"]/(1-params["theta_n"]))

V1_guess = inventory_grid

firm_choices(inventory_grid, p_guess, V1_guess, EV0_guess, params)








def firm_value_function():
return(0)
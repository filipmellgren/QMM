# Firm choices functions
# The innermost loop
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import UnivariateSpline

def firm_choices(inventory_grid, price_guess, EV0_guess, params):
	''' Outputs vectors for each inventory on grid.
	Innermost loop.
	Called by: firm_vf.iterate_firm_vf()
	'''
	sub_period_prod_star, V1_guess = optimal_sub_period_prod(EV0_guess, price_guess, inventory_grid, params) # m_star
	V1_spline = UnivariateSpline(inventory_grid, V1_guess)
	inventory_star, adjust_value = optimal_inventory(inventory_grid, V1_guess, price_guess, params, V1_spline) # s_star
	
	return(inventory_star, adjust_value, sub_period_prod_star, V1_guess)

def optimal_inventory(inventory_grid, V1_guess, price_guess, params, V1_spline):
	''' Find s_star 
	See 1.4.1 page 3 of problem set
	inventory_grid should be a vector
	V1_guess should also be a vector correpsonding to one value per inventory level
	
	'''
	q = params["q"]
	
	inventory_star_sol = minimize_scalar(lambda x : -(V1_spline(x) - price_guess * q * x), method = 'bounded', bounds = (inventory_grid[0], inventory_grid[-1]))
	inventory_star = inventory_star_sol.x
	
	
	adjust_value = -price_guess * q * inventory_star + V1_spline(inventory_star)
	return(inventory_star, adjust_value)

def optimal_sub_period_prod(EV0_guess, price_guess, inventory_grid, params):
	''' Find m_star
	Takes inventory vector, and outputs optimal sub period production, m, for each value of inventory.
	
	'''
	eta = params["eta"]
	beta = params["beta"]
	omega = params["wage"]
	
	m_star = []
	V1_value = []
	
	for ix in range(inventory_grid.shape[0]):
		inv = inventory_grid[ix]
	
		res = minimize_scalar(V1_fun, args=(price_guess, inv, inventory_grid, EV0_guess, params), method='bounded', bounds=(0, inv))
		m_star.append(res.x)
		V1_value.append(-res.fun) # Because minimizing above, there is a negative sign in the function
	
	m_star = np.asarray(m_star)
	V1_value = np.asarray(V1_value)

	n_s = labor_choice_final(inventory_grid, price_guess, params)

	V_corner = price_guess * (final_good_production(inventory_grid, n_s, params) - omega * n_s) + beta * EV0_guess(0) # Eq 10
	V_mat = np.array([V1_value, V_corner])
	corner_sol = np.argmax(V_mat, axis = 0)

	sub_period_production = m_star * (1 - corner_sol) + inventory_grid * corner_sol
	V1 = V1_value * (1 - corner_sol) + V_corner * corner_sol
	
	return(sub_period_production, V1)

def V1_fun(m, price_guess, inventory_level, inventory_grid, EV0_guess, params):
	''' Eq 9
	Same as Tom!
	'''
	sigma = params["sigma"]
	eta = params["eta"]
	beta = params["beta"]
	omega = params["wage"]
	
	n = labor_choice_final(m, price_guess, params)
	inv_next = inventory_level - m
	V1 = price_guess * (final_good_production(m ,n, params) - sigma * (inventory_level - m) - omega * n) + beta * EV0_guess(inv_next)
	return(-V1) # Note for minimzer, switch sign

def intermediate_good_price(price_guess, params):
	''' Return the price of the intermediate good, q.
	See page 3 equation 3 of problem set.
	'''
	alpha = params["alpha"]
	zbar = params["zbar"]
	eta = params["eta"]
	beta = params["beta"]
	delta = params["delta"]
	q = (price_guess**(alpha - 1) /zbar) * ((1 - beta * (1-delta))/(beta * alpha))**alpha * (eta/(1-alpha))**(1-alpha)
	return(q)

def labor_choice_final(m, price_guess, params):
	''' Equation 8 in problem set.
	The choice of n is a static problem, n(m) satisfies wage = G(m,n(m)).
	This gives the labor chosen for final good production.
	'''
	theta_n = params["theta_n"]
	theta_m = params["theta_m"]
	eta = params["eta"]
	return(((theta_n * price_guess * m**theta_m) / eta)**(1/(1-theta_n)))


def final_good_production(m,n, params):
	''' G function
	Also used in market clearing
	'''
	theta_n = params["theta_n"]
	theta_m = params["theta_m"]
	return(m**theta_m * n**theta_n)
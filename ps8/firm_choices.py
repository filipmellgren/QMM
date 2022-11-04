# Firm choices functions
# The innermost loop
import numpy as np
from scipy.optimize import minimize_scalar
import ipdb

def firm_choices(inventory_grid, price_guess, V1_guess, EV0_guess, params):
	''' Outputs vectors for each inventory on grid.
	Innermost loop.
	'''
	inventory_star, adjust_value = optimal_inventory(inventory_grid, V1_guess, price_guess, params) # s_star

	sub_period_prod_star = optimal_sub_period_prod(EV0_guess, price_guess, inventory_grid, params) # m_star
	return(inventory_star, adjust_value, sub_period_prod_star)

def optimal_inventory(inventory_grid, V1_guess, price_guess, params):
	''' Find s_star 
	See 1.4.1 page 3 of problem set
	inventory_grid should be a vector
	V1_guess should also be a vector correpsonding to one value per inventory level
	'''
	q = intermediate_good_price(price_guess, params)
	inventory_star_ix = np.argmax(-price_guess * q * inventory_grid + V1_guess)
	inventory_star = inventory_grid[inventory_star_ix]
	adjust_value = -price_guess * q * inventory_star + V1_guess[inventory_star_ix]
	return(inventory_star, adjust_value)

def optimal_sub_period_prod(EV0_guess, price_guess, inventory_grid, params):
	''' Find m_star
	Takes inventory vector, and outputs optimal sub period production, m, for each value of inventory.
	
	'''
	eta = params["eta"]
	beta = params["beta"]
	omega = eta / price_guess
	
	m_star = []
	V1_value = []
	
	for ix in range(inventory_grid.shape[0]):
		inv = inventory_grid[ix]
	
		res = minimize_scalar(V1_fun, args=(price_guess, inv, inventory_grid, EV0_guess, params), method='bounded', bounds=(0, inv))
		m_star.append(res.x)
		V1_value.append(-res.fun)
	
	m_star = np.asarray(m_star)
	V1_value = np.asarray(V1_value)

	n_s = labor_choice_final(m_star, price_guess, params)

	V_corner = price_guess * (final_good_production(inventory_grid, n_s, params) - omega * n_s) + beta * EV0_guess[0]
	V_mat = np.array([V1_value, V_corner])
	corner_sol = np.argmax(V_mat, axis = 0)

	sub_period_production = m_star * (1 - corner_sol) + inventory_grid * corner_sol
	return(sub_period_production)

def V1_fun(m, price_guess, inventory_level, inventory_grid, EV0_guess, params):
	''' Eq 9
	Same as Tom!
	'''
	sigma = params["sigma"]
	eta = params["eta"]
	beta = params["beta"]
	omega = eta / price_guess
	
	n = labor_choice_final(m, price_guess, params)
	
	next_period_inventory, ix = find_nearest(inventory_level - m, inventory_grid)

	V1 = price_guess * (final_good_production(m ,n, params) - sigma * (inventory_level - m) - omega * n) + beta * EV0_guess[ix] # Note EV0_guess is a value function of inventory levels
	return(-V1) # Note for minimzer, switch sign

def find_nearest(point, grid):
	# TODO: can we do this for vectors as well?
	ix = np.argmin(np.abs(grid - point))
	point_on_grid = grid[ix]
	return(point_on_grid, ix)

def intermediate_good_price(price_guess, params):
	''' Return the price of the intermediate good, q.
	See page 3 equation 3 of problem set.
	'''
	alpha = params["alpha"]
	zbar = params["zbar"]
	eta = params["eta"]
	beta = params["beta"]
	delta = params["delta"]
	q = price_guess**(alpha - 1) /zbar * ((1 - beta * (1-delta))/(beta * alpha))**alpha * (eta/(1-alpha))**(1-alpha)
	return(q)

def labor_choice_final(m, price_guess, params):
	theta_n = params["theta_n"]
	theta_m = params["theta_m"]
	eta = params["eta"]
	return((theta_n * price_guess * m**theta_m) / eta)**(1/(1-theta_n))


def final_good_production(m,n, params):
	''' G function
	'''
	theta_n = params["theta_n"]
	theta_m = params["theta_m"]
	return(m**theta_m * n**theta_n)
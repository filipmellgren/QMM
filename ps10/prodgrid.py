
import numpy as np
from scipy.stats import pareto

def gen_prod_grid(alpha, gamma):
	''' Generate a productivity shock grid.
	This grid is defined as the expected value between two pareto distributed endpoints (as found in grid_endp). Probability to end up on a point in the prod_grid is the cdf of ending up at the grid endpoint - the cdf of ending up at the lower endpoint. 

	Small firms are appended to the grid. They are asumed to have an "average productitivty". A firm is small with probability 80%.
	'''
	pct80 = pareto.ppf(0.8, alpha)
	pct9999 = pareto.ppf(0.9999, alpha)

	grid_endp = np.logspace(start = np.log(pct80), stop = np.log(pct9999), num = 11, base = np.exp(1))

	prod_grid = (alpha /(-alpha + gamma -1) * (grid_endp[1:]**(-alpha + gamma - 1) - grid_endp[0:-1]**(-alpha+gamma-1)) / (grid_endp[0:-1]**(-alpha) - grid_endp[1:]**(-alpha)) )**(1/(gamma - 1)) # Slide 17, lecture 2

	small_firm_productivity = (alpha / (-alpha + gamma - 1) * (pct80**(-alpha + gamma - 1) - 1) / (1-pct80**(-alpha)) )**(1/(gamma - 1))

	prod_grid = np.insert(prod_grid, 0, small_firm_productivity)

	# Probabilities
	probs = pareto.cdf(grid_endp[1:], alpha) - pareto.cdf(grid_endp[0:-1], alpha)
	probs = probs/0.9999
	probs = np.insert(probs, 0, 1-np.sum(probs))

	return(prod_grid, probs)

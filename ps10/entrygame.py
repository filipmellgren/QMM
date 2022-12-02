# QMM2, Josh's problem set
import numpy as np
from scipy.stats import pareto
from scipy import optimize
import time
import scipy as sp

import ipdb
def agg_equilibrium(c_curvature, c_shifter, z_draws, n_industries, prod_grid, W):
	''' Computes an aggregate equilibrium for the entry game economy. Returns cost weighted variables by each percentile
	TODO: not yet in production.
	'''
	# Industry variables, arrays with one value per industry
	shares, costs, entry_costs, hhis, markups, prices, prices_j = [], [], [], [], [], [], []
	for industry in range(n_industries):
		z_draw = z_draws[industry, :]
		share, cost, entry_cost, hhi, markup, price, Pj = industry_equilibrium(c_curvature, c_shifter, z_draw, prod_grid, W) # TODO: my costs are marginal costs. Not toal costs
		shares.append(share), 
		costs.append(cost), entry_costs.append(entry_cost)
		hhis.append(hhi), markups.append(hhi)
		prices.append(price), prices_j.append(Pj)

	ipdb.set_trace()
	Z = aggregate_Z(z_draws, prices, prices_j, gamma, theta)
	Ptilde = price_aggregator(prices_j, theta)
	W = (1/Ptilde)**(1/(1-theta))
	Y = W**theta
	entry_cost = np.mean(entry_cost)
	L = (Y - entry_costs)/Z

	share_pcts, cost_pcts = [], []
	for pct in range(100):
		share_pct, cost_pct = np.percentile(shares, pct), np.percentile(costs, pct)
		share_pcts.append(share_pct), cost_pcts.append(cost_pct)

	shares_cw, hhis_cw, markups_cw, prices_cw = [], [], [], []
	for pct in range(99):
		share_cw = weighted_avg(cost_pcts[pct], cost_pcts[pct + 1], share_pcts[pct], share_pcts[pct + 1])
		hhi_cw, markup_cw, price_cw = 1,1,1 # TODO: repeat above weighted average?
		shares_cw.append(share_cw), hhis_cw.append(hhi_cw), markups_cw.append(markups_cw), prices_cw.append(price_cw)
	
	return(shares_cw, hhis_cw, markups_cw, prices_cw)

def aggregate_Z(firm_prods, firm_prices, Pj, gamma, theta):
	Zj = np.mean(firm_prods**(-1) * price**(-gamma), axis = 1) # TODO: check what axis to use
	assert Zj.size == 1000, "Seems like axis should be changed in aggregae_prod"
	Z = (np.mean(Zj * Pj**(gamma - theta)))**(-1)
	return(Z)

def weighted_avg(w1, w2, v1, v2):
	return((w1*v1 + w2*v2) /(w1 + w2))

def entry_decisions(n, c_curvature, c_shifter, z_draws, prod_grid, probs, W, gamma, theta):
	''' Calculate net benefit to marginal enmtrant
	Find equilibrium. Probably unique ^_^.

	INPUT
	z_draws : is the incumbent productivity draws
	'''
	ipdb.set_trace()	
	z_draws_incumbent = z_draws[:int(n)]
	
	entry_costs = W * c_shifter * (n+1) ** c_curvature
	
	enter_profits = []

	for z_entrant in prod_grid:
		z_draws_test = np.append(z_draws_incumbent, z_entrant)
		profits, _, _,_ = get_profits(W, z_draws_test, gamma, theta)
		enter_profits.append(profits[-1])

	value_entering = enter_profits @ probs

	net_benefit = value_entering - entry_costs

	return(net_benefit, entry_costs)

def get_profits(W, z_draws, gamma, theta):
	''' Calculate a profit vector of size n

	Calculates profits in an inudstry.
	Demand is nested CES from perfectly competitive final good producers.
	'''

	#Yj = np.sum(y**((gamma-1)/gamma))**(gamma/(gamma - 1))
	#Y = np.sum(Yj**((theta-1)/(theta)))**(theta/(theta - 1))
	Y = W**theta # NOTE: wortks for initial steady state. Just how we normalize prices and units allowing this simplification. 

	markup_guess = np.repeat(gamma / (gamma -1), repeats = np.size(z_draws))

	# TODO: Cournot competition is assumed. Note, they are strategic complements, so algorithm might not work for that reason.

	# TODO doesn't work, code his newton by hand. (it probably guesses on negative markups)
	#markups = optimize.newton(lambda x : markup_diff(x, z_draws, gamma, theta)[0], guess)
	learning_rate = 1
	tol = 1e-4
	
	markups = markup_optim(markup_guess, learning_rate, z_draws, gamma, theta, tol)
	
	pij = (W * markups / z_draws)

	Pj = price_aggregator(pij, gamma) # Scales with W

	profits = (1 - 1/markups) * (markups * W / z_draws)**(1-gamma) * Pj**(gamma - theta) * Y # Scales with W**(1-theta)*Y

	return(profits, markups, pij, Pj)

def markup_optim(guess, learning_rate,  z_draws, gamma, theta, tol):
	''' Find industry markups using Newton's method
	slide 27
	'''
	Delta = learning_rate
	error = tol + 10
	while np.any(error > tol): # TODO: Might be expensive to check entire vector, all the time

		error, markups, shares, elas, elas_grad = markup_diff(guess, z_draws, gamma, theta)
		grad = 1 + 1/((elas-1)**2) * elas_grad * (1-gamma)*shares/guess
		guess = guess - Delta * error / grad
	
	return(guess)


def markup_diff(markup_guess, z_draws, gamma, theta):
	''' Calculates difference betwee markup guess and implied markups
	TODO: change name to indsutry_eqlbrm?
	Also returns markups, shares, and elasticities used along the way
	'''
	
	shares = ((markup_guess/z_draws)**(1-gamma))/(np.sum((markup_guess/z_draws)**(1-gamma)))
	#elas_c = ((1 - shares)*gamma**(-1) + shares*theta**(-1))**(-1)# Elasticity under cournot
	elas_b = (1 - shares)*gamma + shares * theta # Elasticity under Bertrand, slide 22
	elas_grad = -gamma + theta
	markups = elas_b/(elas_b - 1)

	loss = markup_guess - markups

	return(loss, markups, shares, elas_b, elas_grad)

def price_aggregator(p, elas):
	''' Price index calculation

	p : is a n input vector of lower level prices
	elas : is the within aggregation unit elasticity of substitution
	'''
	P = (np.sum(p**(1-elas)))**(1/(1-elas))
	return(P)

def industry_equilibrium(c_curvature, c_shifter, z_draws, prod_grid, W):
	''' Simulate an industr, given prodcutivity draws.

	Two stage game based on Edmond, Midrigan, Xu (2022)
		1st period: Firms choose whether to pay an entry cost and enter
		2nd period: Static nested CES model

	'''
	gamma = 10.5
	theta = 1.24

	# First stage: Entry decisions
	# TODO: if doesn't workj, it is becasue it returns two things, not just one. Use lambda funciton instead. 
	n_incumbents = optimize.bisect(entry_decisions, 1, 5000-1, args = (c_curvature, c_shifter, z_draws, prod_grid, probs, W, gamma, theta), xtol = 0.9) # TODO: check terminal condition

	_, entry_costs = entry_decisions(n_incumbents, c_curvature, c_shifter, z_draws, prod_grid, probs, W, gamma, theta)

	marginal_entrant = n_incumbents + 1
	# Second stage: Static nested CES from lecture 1
	
	z_draws = z_draws[:int(marginal_entrant)] # TODO: return z_drawsd from entry decions
	_, markups, prices, Pj = get_profits(W, z_draws, gamma, theta)
	
	loss, _, shares, elas,_ = markup_diff(markups, z_draws, gamma, theta) 

	assert np.all(loss < 1e-4), "Markup loss might be too high."

	largest_firm = np.argmax(shares)
	share = shares[largest_firm]
	hhi = np.sum(np.asarray(shares)**2)

	cost = prices[largest_firm]*(1-markups[largest_firm])

	return(share, cost, entry_costs, hhi, markups, prices, Pj)

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


alpha = 6
gamma = 10.5
prod_grid, probs = gen_prod_grid(alpha, gamma)

n_industries = 1000
n_industries = 2
z_draws = np.random.choice(prod_grid, size = (n_industries, 5000), p = probs)
# Note on W: suppose W**(-theta)Y = 1. Calcualte equilibrium as if W and Y are 1.
# Compute final good price, then use PW = 1, set Y st W**(-theta)Y = 1. 
# Finally, set L equal to aggregate labor used across industries by firms for proeuction and entry costs. 
# To 
W = 1
c_curvature = 0
c_shifter = 2e-4

start = time.time()
parta = agg_equilibrium(c_curvature, c_shifter, z_draws, n_industries, prod_grid, W)
end = time.time()


0.94 * prod_grid@probs

partb = agg_equilibrium(1, 2e-7, z_draws, n_industries)



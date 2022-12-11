# Industry equilibrium with killer acquisitions and not taking w**theta = Y = 1 as given
# This file assumes there exists an L_supply that can be obtained from np.loadtxt("total_labor.txt"). Running entrygame.py first creates that file. 

# TODO: This shouild be merged with industry_eq.py. Functions will need to be more general first. Currently incompatible as industry_eq.py is mostly based on taking W and Y as given whereas this file iterates over W**-theta * Y


import numpy as np
from scipy.stats import pareto
from scipy import optimize
import time
import scipy as sp
from prodgrid import gen_prod_grid
from industry_eq import industry_equilibrium, price_aggregator, get_entry_costs
from entrygame import find_wage, agg_variables
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"

import pandas as pd


def entry_decisions(n, ratio, c_curvature, c_shifter, z_draws, prod_grid, probs, gamma, theta, competition, learning_rate):
	''' Calculate net benefit to marginal entrant
	Find equilibrium. Probably unique ^_^.
	Compared against net benefit for ratio = W = Y = 1 and it gives the same value for number of incumbents. Add one for marginal entrant

	See slide 19 L2 on how the state is reduced

	TODO: aggregate small firms to one large firm Slide 15 L2 for a speed gain. 
	
	z_draws : is the incumbent productivity draws
	killer_acq : Is an indicator whether the first firm is allowed to acquire to kill
	ratio : is W**(-theta) * Y which is what entry decisions ultimately depend on
	'''	
	z_draws_incumbent = z_draws[:int(n)]
	
	enter_values_list = []

	for z_entrant in prod_grid:
		# Calc value proportional to profits
		z_draws_entrant = np.append(z_draws_incumbent, z_entrant)
		markups = get_markups(gamma, theta, z_draws_entrant, competition)
	
		Pj_divW = price_aggregator(markups / z_draws_entrant, gamma) # Should be multiplied by W
		
		enter_values = (1 - 1/markups) * (markups / z_draws_entrant)**(1-gamma) * Pj_divW**(gamma - theta) 
		enter_values_list.append(enter_values[-1])

	value_entering = np.array(enter_values_list) @ probs

	eps = c_shifter * (n+1) ** c_curvature
	decision = ratio * value_entering/eps -1 # We look for the root value and decision is enter if positive, otherwise not. 
	return(decision)

def get_markups(gamma, theta, z_draws, competition):
	''' Calculate markups given paramters, productivity, and type of compeition
	competition is either "Bertrand" or "Cournot".
	'''
	markup_guess = np.repeat(gamma / (gamma -1), repeats = np.size(z_draws))
	learning_rate = 0.5
	tol = 1e-7
	markups = markup_optim(markup_guess, learning_rate, z_draws, gamma, theta, competition, tol)
	return(markups)

def markup_optim(guess, learning_rate,  z_draws, gamma, theta, competition, tol):
	''' Find industry markups using Newton's method
	slide 27
	Markups here simply average cost
	NOTE: this seems to generate cycles depending on learning rate. For Cournot, Delta =1 might be too much.
	'''
	Delta = learning_rate
	error = tol + 10
	while error > tol: 
		error, markups, shares, elas, elas_grad = markup_diff(guess, z_draws, gamma, theta, competition)
		grad = 1 + ( 1/((elas-1)**2) ) * elas_grad * (1-gamma)*shares/guess
		guess = guess - Delta * error / grad
		error = np.sqrt(np.sum(error**2))
		
	return(guess)

def markup_diff(markup_guess, z_draws, gamma, theta, competition):
	''' Calculates difference betwee markup guess and implied markups
	
	Also returns markups, shares, and elasticities used along the way
	'''

	shares = ((markup_guess/z_draws)**(1-gamma))/(np.sum((markup_guess/z_draws)**(1-gamma)))
	
	if competition == "Bertrand":
		elas = (1 - shares)*gamma + shares * theta # Elasticity under Bertrand, slide 22, lecture 1
		elas_grad = -gamma + theta
	if competition == "Cournot":
		# Cournot seems less stable and leads to "overflow encountered in reduce"
		elas = ((1 - shares)*gamma**(-1) + shares*theta**(-1))**(-1)# Elasticity under cournot
		elas_grad = (1/theta - 1/gamma)/(( (1-shares) / gamma + shares / theta )**2)# For Cournot

	markups = elas/(elas - 1)

	loss = markup_guess - markups

	return(loss, markups, shares, elas, elas_grad)

def industry_equilibrium(ratio, c_curvature, c_shifter, z_draws, prod_grid, probs, gamma, theta, competition, learning_rate):
	''' Simulate an industr, given prodcutivity draws.
	
	This implementation does not assume 
	Three stage game based on Edmond, Midrigan, Xu (2022), with a killer acquisition phase added
		0th period: Incumbent decides whether to acquire potential entrant
		1st period: Firms choose whether to pay an entry cost and enter
		2nd period: Static nested CES model
	
	INPUT :
	ratio : W**(-theta) * Y a guess of what this is. Want supply = demand so find this using bisection. Note, these are aggregates so find them in aggregate equilbirium.
	
	OUTPUT : 
	'''
 
	# First stage: Entry decisions
	# TODO: incumbent needs to make its decision before we do this
	try:
		n_incumbents = optimize.bisect(lambda x: entry_decisions(x, ratio, c_curvature, c_shifter, z_draws, prod_grid, probs, gamma, theta, competition, learning_rate), 1, z_draws.size-1, xtol = 0.9)

	except ValueError:
		high_point = entry_decisions(1, ratio, c_curvature, c_shifter, z_draws, prod_grid, probs, gamma, theta, competition, learning_rate)
		low_point = entry_decisions(1, ratio, c_curvature, c_shifter, z_draws, prod_grid, probs, gamma, theta, competition, learning_rate)
		if high_point > 0 and low_point > 0:
			n_incumbents = z_draws.size -1
		if high_point < 0 and low_point < 0:
			n_incumbents = 1
		
	marginal_entrant = n_incumbents + 1

	z_draws = z_draws[:int(marginal_entrant)]

	# Second stage: Static nested CES from lecture 1
	
	markups = get_markups(gamma, theta, z_draws, competition)
	
	Ptildej = price_aggregator(markups/z_draws, gamma) # Slide 42 L1

	markupj = (Ptildej**(1 - gamma)) / np.sum(markups ** (-gamma) * z_draws**(gamma - 1))
	
	# Slide 45 lecture 1
	Zj = (np.sum((markups/markupj)**(-gamma) * z_draws**(gamma - 1)))**(1/(gamma - 1))

	_, _, shares, _, _ = markup_diff(markups, z_draws, gamma, theta, competition)
	share_large = shares[0]
	profit_constant_large = (1-1/markups[0]) * share_large # Multiply by total output Y

	return(markupj, Ptildej, Zj, int(marginal_entrant), profit_constant_large, share_large)

def agg_mc(ratio, L_supply, c_curvature, c_shifter, z_draws, prod_grid, probs, gamma, theta, competition, learning_rate, path):
	''' Checks aggregate market clearing. 
	Use only after solving for an initial equilibrium, as it depends on L_supply which is only known after it has been solved for in equilibrium once. 
	ratio: is the W**(-theta) Y value using which firms decide whether to enter or not. It reduces the state space.

	For ratio = 1 = W = Y, L becomes the same as in Exc1 part b
	
	'''
	# Industry variables, arrays with one value per industry
	markups, prices_j, Z_js, marginal_entrants = [], [], [], []
	profits_large, shares_large = [], []

	for industry in range(z_draws.shape[0]):	
		z_draw = z_draws[industry, :]
		markupj, P_tildej, Zj, marginal_entrant, profit_large, share_large = industry_equilibrium(ratio, c_curvature, c_shifter, z_draw, prod_grid, probs, gamma, theta, competition, learning_rate) 

		markups.append(markupj)
		prices_j.append(P_tildej)
		Z_js.append(Zj)
		marginal_entrants.append(marginal_entrant)
		profits_large.append(profit_large)
		shares_large.append(share_large)
	
	
	P = 1 # Given by problem set

	Lprod, L, W_implied, Y_tilde, Z, Markup, costj, avg_firms = agg_variables(ratio, P, np.array(P_tildej), np.array(Z_js), np.array(markups), marginal_entrants, theta, c_shifter, c_curvature)
		
	share_large_avg = np.mean(np.array(shares_large))
	profit_large_avg = np.mean(np.array(profits_large) * Y_tilde)
	loss = L - L_supply
	markups = np.array(markups)
	
	return(loss, markups, np.array(prices_j), np.array(Z_js), Y_tilde, W_implied, avg_firms, profit_large_avg, share_large_avg, P)

def cost_from_prices(Zj, Pj, P, W):
	''' Obtain firm marginal cost from productivity, prices, and wage
	First relationship – Slide: 19, L1
	Second relationship – Slide 46, L1 + using relationship of industry markup to find expression for aggregate markup.
	'''
	Yj = (Pj/P)**(-theta) * Y 
	costj = Yj / Zj
	return(costj)

def agg_equilibrium(L_supply, c_curvature, c_shifter, z_draws, prod_grid, probs, gamma, theta, competition, learning_rate, path):
	''' Computes an aggregate equilibrium for the entry game economy. Returns cost weighted variables by each percentile

	# TODO: add case where W=Y = 1
	'''
	
	X = optimize.bisect(lambda x: agg_mc(x, L_supply, c_curvature, c_shifter, z_draws, prod_grid, probs, gamma, theta, competition, learning_rate, path)[0], 1e-1, 1e+1, xtol = 1e-5)
	
	_, markups, prices_j, Z_js, Y, W, avg_firms, profit_large, share_large, P = agg_mc(X, L_supply, c_curvature, c_shifter, z_draws, prod_grid, probs, gamma, theta, competition, learning_rate, path)

	hh_welfare = get_hh_welfare(W, L_supply, P)

	return(profit_large, share_large, avg_firms, hh_welfare, W, Y, prices_j)

def get_hh_welfare(W, L_supply, P):
	''' Returns HH welfare
	Use BC and FOC on slide 40 to back out parameter psi, assuming rho = 2
	'''
	hh_c = W * L_supply / P # From BC
	rho = 2 # Assumption. TODO: make consistent with curvatures and shift?
	psi = -1/rho * np.log(hh_c * W / P) / np.log(L_supply) # Rewrite FOC
	hh_welfare = (hh_c**(1-rho)-1)/(1-rho) - 1/(psi + 1) * L_supply**(psi + 1)
	return(hh_welfare)

alpha = 6
gamma = 10.5
theta = 1.24
prod_grid, probs = gen_prod_grid(alpha, gamma)
z_large = prod_grid[-1]

z_draws = np.repeat(prod_grid[0], 5000-1)

z_draws = np.insert(z_draws, 0, z_large)
z_draws = np.expand_dims(z_draws, 0)

prod_grid = np.expand_dims(prod_grid[0], axis = 0)
probs = np.expand_dims(1, axis = 0)

c_curvature = 1
c_shifter = 2e-7
L_supply = np.loadtxt("total_labor.txt")

learning_rate = 1

profit_large, share_large, avg_firms, hh_welfare, W, Y, Pj = agg_equilibrium(L_supply, c_curvature, c_shifter, z_draws, prod_grid, probs, gamma, theta, "Bertrand", learning_rate, "figures/scrap")

f = open("killer_acq.txt", "w")
f.write("No KA " + "profit_large:" + str(profit_large) + ", Share:" + str(share_large) + ", avg firms" + str(avg_firms) + ", hh welfare: " + str(hh_welfare))
f.close()

def agg_eq_ka(k, firms_no_ka, profit_no_ka, gamma, theta, c_shifter, c_curvature):
	''' Computes aggregate equilibrium difference in profits when the initial firm acquires all firms following the kth entrant.

	Acquiring the kth entrant and everyoen afterwards (due to increasing costs to enter and same effect on profit_large as kth entrant) can be both profitable and not profitable.

	This function tells whether it is profitable to acquire everyuione following the kth entrant. 

	A key assumption is that the profit of the future entrants remains at the same constant value. Otherwise, an effect of acquiring a firm is that the future profits of potential entrants increases, and therefore the price for the incumbent. 
	TODO: should I change Pj to 1? 
	'''
	print(k)
	# Effect on profit_large comes from limiting number of entrants to k.
	# Everyone after k is acquired as cost of acquiring additional firms decrease while effect on profits is constant. 
	z_draws_limited = z_draws[:, 0:k]

	profit_large, share_large, avg_firms, hh_welfare, W, Y, Pj = agg_equilibrium(L_supply, c_curvature, c_shifter, z_draws_limited, prod_grid, probs, gamma, theta, "Bertrand", learning_rate, "figures/scrap")

	markup_s = gamma / (gamma - 1)
	Pj = 1 # TODO: think this normalization should be made as we know P = 1,
	# And there is only one industry (for problem set), so Pj = 1. 
	
	entry_profits = (1 - 1 / markup_s) * (markup_s * W / z_draws[:, k:firms_no_ka])**(1 - gamma) * Pj**(gamma - theta) * Y
	
	entry_costs = get_entry_costs(W, c_shifter, np.arange(k+1, firms_no_ka+1), c_curvature)

	acq_price = np.sum(entry_profits - entry_costs)

	net_profit_large = profit_large - acq_price
	
	return(profit_no_ka - net_profit_large)
	
from scipy.optimize import minimize_scalar

sol = minimize_scalar(lambda x: agg_eq_ka(int(x), int(avg_firms), profit_large, gamma, theta, c_shifter, c_curvature), bounds = (2, avg_firms), method = "bounded", options = {"xatol" : 0.9})

z_draws = np.repeat(prod_grid[0], int(sol.x))

z_draws = np.insert(z_draws, 0, z_large)
z_draws = np.expand_dims(z_draws, 0)

profit_large, share_large, avg_firms, hh_welfare, W, Y, Pj = agg_equilibrium(L_supply, c_curvature, c_shifter, z_draws, prod_grid, probs, gamma, theta, "Bertrand", learning_rate, "figures/scrap")

f = open("killer_acq.txt", "a")
f.write("\n KA " + "profit_large:" + str(profit_large) + ", Share:" + str(share_large) + ", avg firms" + str(avg_firms) + ", hh welfare: " + str(hh_welfare))
f.close()










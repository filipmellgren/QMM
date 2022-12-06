import numpy as np
from scipy import optimize
import ipdb

def entry_decisions(n, c_curvature, c_shifter, z_draws, prod_grid, probs, W, gamma, theta, Y, competition, learning_rate):
	''' Calculate net benefit to marginal enmtrant
	Find equilibrium. Probably unique ^_^.

	TODO: aggregate small firms to one large firm Slide 15 L2

	INPUT
	z_draws : is the incumbent productivity draws
	'''
		
	z_draws_incumbent = z_draws[:int(n)]
	
	entry_costs = W * c_shifter * (n+1) ** c_curvature
	
	enter_profits = []

	for z_entrant in prod_grid:
		
		z_draws_test = np.append(z_draws_incumbent, z_entrant)
		profits, _, _,_ = get_profits(W, z_draws_test, gamma, theta, Y, competition,learning_rate)
		enter_profits.append(profits[-1])

	value_entering = enter_profits @ probs

	net_benefit = value_entering - entry_costs

	return(net_benefit, entry_costs)

def get_profits(W, z_draws, gamma, theta, Y, competition, learning_rate):
	''' Calculate a profit vector of size n

	Calculates profits in an industry as if W**theta = Y = 1.

	Demand is nested CES from perfectly competitive final good producers.

	'''

	markup_guess = np.repeat(gamma / (gamma -1), repeats = np.size(z_draws))

	learning_rate = 0.5
	tol = 1e-7
	
	markups = markup_optim(markup_guess, learning_rate, z_draws, gamma, theta, competition, tol)
	
	pij = (W * markups / z_draws) # Slide 42, L1: assuming W = 1. TODO: better source

	Pj = price_aggregator(pij, gamma) # Scales with W (note, not final as we compute industry *as if* each of the wage and Y = 1.

	profits = (1 - 1/markups) * (markups * W / z_draws)**(1-gamma) * Pj**(gamma - theta) * Y # Scales with W**(1-theta)*Y. Slide: 15 Lecture 2 (small firms) TODO Different for large firms?

	return(profits, markups, pij, Pj)

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

def price_aggregator(p, elas):
	''' Price index calculation
	Slide 19 lecture 1 

	p : is a n input vector of lower level prices
	elas : is the within aggregation unit elasticity of substitution
	'''
	P = (np.sum(p**(1-elas)))**(1/(1-elas))
	#P = (np.mean(p**(1-elas)))**(1/(1-elas)) # TODO: which one?
	return(P)

def industry_equilibrium(c_curvature, c_shifter, z_draws, prod_grid, probs, W, gamma, theta, competition, learning_rate):
	''' Simulate an industr, given prodcutivity draws.

	Two stage game based on Edmond, Midrigan, Xu (2022)
		1st period: Firms choose whether to pay an entry cost and enter
		2nd period: Static nested CES model

	TODO: what cost should I return?
		- the cost is used to weigh the value of an industry when computing outcome variables between percentiles.
		- Tom suggested using an average within industry
	# Industry cost, or largest firm cost?
	# Variabel or total?
	'''

	Y = W**theta # NOTE: wortks for initial steady state. Just how we normalize prices and units allowing this simplification. 
	# First stage: Entry decisions

	n_incumbents = optimize.bisect(lambda x: entry_decisions(x, c_curvature, c_shifter, z_draws, prod_grid, probs, W, gamma, theta, Y, competition, learning_rate)[0], 1, 5000-1, xtol = 0.9) # TODO: check terminal condition
	
	marginal_entrant = n_incumbents + 1

	# Second stage: Static nested CES from lecture 1
	z_draws = z_draws[:int(marginal_entrant)]
	_, markups, prices, Pj = get_profits(W, z_draws, gamma, theta, Y, competition, learning_rate)
	loss, _, shares, elas,_ = markup_diff(markups, z_draws, gamma, theta, competition) 

	# Summary variables
	largest_firm = np.argmax(shares)
	share = shares[largest_firm]
	hhi = np.sum(np.asarray(shares)**2)
	Ptildej = price_aggregator(markups/z_draws, gamma) # Slide 42 L1
	# Industry markup Slide 45 lecture 1. "Industry markup is the cost weighted markup"
		# Slide 42 L1, true price index Pj = W * Ptildej -> Pj/W = Ptildej
	markupj = (Ptildej**(1 - gamma)) / np.sum(markups ** (-gamma) * z_draws**(gamma - 1))
	

	# demandij = prices **(-gamma) * Pj**(gamma-theta)*Y# Slide 14, lecture 2. Small firms
	# costs = demandij * prices / markups # TODO quantities Slide 4x lecture 1 ()
	# Slide 45 lecture 1
	Zj = (np.sum((markups/markupj)**(-gamma) * z_draws**(gamma - 1)))**(1/(gamma - 1))

	return(share, hhi, markupj, Ptildej, Zj)



def industry_Zj(z_draws, prices, gamma):
	''' TODO: not sldie 46?
	'''
	Zj = np.mean(z_draws**(-1) * prices**(-gamma))
	return(Zj)
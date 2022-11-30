# QMM2, Josh's problem set
import numpy as np
def agg_equilibrium(c_curvature, c_shifter, z_draws, n_industries):
	''' Computes an aggregate equilibrium for the entry game economy. Returns cost weighted variables by each percentile
	'''

	# Industry variables, arrays with one value per industry
	shares, costs, hhis, markups, prices = [], [], [], [], []
	for industry in range(n_industries):
		z_draw = z_draws[industry, :]
		share, cost, hhi, markup, price = industry_equilibrium(c_curvature, c_shifter, z_draw)
		shares.append(share), costs.append(cost)
		hhis.append(hhi), markups.append(hhi), prices.append(price)

	share_pcts, cost_pcts = [], []
	for pct in range(100):
		share_pct, cost_pct = np.percentile(shares, pct), np.percentile(costs, pct)
		share_pcts.append(share_pct), cost_pcts.append(cost_pct)

	shares_cw, hhis_cw, markups_cw, prices_cw = [], [], [], []
	for pct in range(99):
		share_cw = weighted_avg(cost_pcts[pct], cost_pcts[pct + 1], share_pcts[pct], share_pcts[pct + 1])
		hhi_cw, markup_cw, price_cw = 1,1,1 # TODO: repeat above?
		shares_cw.append(share_cw), hhis_cw.append(hhi_cw), markups_cw.append(markups_cw), prices_cw.append(price_cw)
	
	return(share_cw, hhi_cw, markup_cw, price_cw)

def weighted_avg(w1, w2, v1, v2):
	return((w1*v1 + w2*v2) /(w1 + w2))

def entry_decisions(c_curvature, c_shifter, z_draws):

	entry_costs = W * eps * n ** cost_curvature
	entry_benefits = 1 # TODO conditional on no future entry
	entries = entry_benefits > entry_costs

	return(entries)

def static_nested_CES(entries, z_draws):

	pass

def industry_equilibrium(c_curvature, c_shifter, z_draws):
	''' Simulate an industr, given prodcutivity draws.

	Two stage game based on Edmond, Midrigan, Xu (2022)
		1st period: Firms choose whether to pay an entry cost and enter
		2nd period: Static nested CES model

	'''
	gamma = 10.5
	theta = 1.24

	# First stage: Entry decisions
	entries = entry_decisions(c_curvature, c_shifter)

	# Second stage: Static nested CES from lecture 1
	shares, price, costs = static_nested_CES()

	shares = [0.6, 0.4]

	share = np.max(shares)
	hhi = np.sum(np.asarray(shares)**2)

	cost = 0.5	
	price = 1
	markup = (price - cost)/price
	return(share, cost, hhi, markup, price)

alpha = 6
n_industries = 1000
z_draws = np.random.pareto(alpha, (n_industries, 5000))

z_draws[2, :].shape

parta = agg_equilibrium(0, 2*e-4, z_draws, n_industries)
partb = agg_equilibrium(1, 2*e-7, z_draws, n_industries)


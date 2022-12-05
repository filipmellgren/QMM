# QMM2, Josh's problem set
import numpy as np
from scipy.stats import pareto
from scipy import optimize
import time
import scipy as sp
from prodgrid import gen_prod_grid
from industry_eq import industry_equilibrium


import pandas as pd

import ipdb
def agg_equilibrium(c_curvature, c_shifter, z_draws, n_industries, prod_grid, probs, W, gamma, theta, competition, learning_rate):
	''' Computes an aggregate equilibrium for the entry game economy. Returns cost weighted variables by each percentile
	TODO: how to cost weight?
	Tom said that we find the percentiles, then compute an average over all industries within that range.
	'''
	# Industry variables, arrays with one value per industry
	shares, costs, entry_costs, hhis, markups, prices, prices_j = [], [], [], [], [], [], []
	Z_js = []
	
	for industry in range(n_industries):
		z_draw = z_draws[industry, :]
		share, hhi, markupj, Pj, Zj, cost = industry_equilibrium(c_curvature, c_shifter, z_draw, prod_grid, probs, W, gamma, theta, competition, learning_rate) 

		shares.append(share), 
		costs.append(cost)#, entry_costs.append(entry_cost)
		hhis.append(hhi), markups.append(markupj)
		#prices.append(price), 
		prices_j.append(Pj)
		Z_js.append(Zj)
	
	ipdb.set_trace()

	share_pcts, cost_pcts = [], []
	for pct in range(100):
		share_pct, cost_pct = np.percentile(shares, pct), np.percentile(costs, pct)
		share_pcts.append(share_pct), cost_pcts.append(cost_pct)

	ipdb.set_trace()
	Z = aggregate_Z(np.array(Z_js), np.array(prices_j), gamma, theta)
	Ptilde = price_aggregator(np.array(prices_j), theta)
	W = (1/Ptilde)**(1/(1-theta))
	Y = W**theta
	entry_cost = np.mean(entry_costs)
	L = (Y - entry_cost)/Z


	shares_cw, hhis_cw, markups_cw, prices_cw = [], [], [], []
	for pct in range(99):
		share_cw = weighted_avg(cost_pcts[pct], cost_pcts[pct + 1], share_pcts[pct], share_pcts[pct + 1])
		hhi_cw, markup_cw, price_cw = 1,1,1 # TODO: repeat above weighted average?
		shares_cw.append(share_cw), hhis_cw.append(hhi_cw), markups_cw.append(markups_cw), prices_cw.append(price_cw)
	
	return(shares_cw, hhis_cw, markups_cw, prices_cw)



def aggregate_Z(Zj, markupj, markup, theta):
	''' Slide 46, lecture 1
	Pj, Zj : arrays of values for all industries, each j in J.
	'''
	#Z = (np.mean(Zj * Pj**(gamma - theta)))**(-1)
	markup = Y / (W * L) # TODO What is this?
	Z = (np.mean((markupj/markup)**(-theta) * Zj**(theta-1)))**(1/(theta - 1))
	return(Z)

def weighted_avg(w1, w2, v1, v2):
	return((w1*v1 + w2*v2) /(w1 + w2))

alpha = 6
gamma = 10.5
theta = 1.24
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


#z_draw = z_draws[0]
#pd.DataFrame(z_draw).to_csv("z.csv", index = False)
#z_draw = np.array(pd.read_csv("Prod_vec.csv"))
#z_draw = np.squeeze(z_draw, axis = 1)

parta = agg_equilibrium(c_curvature, c_shifter, z_draws, n_industries, prod_grid, probs, W, gamma, theta, "Bertrand", 0.5)


0.94 * prod_grid@probs

partb = agg_equilibrium(1, 2e-7, z_draws, n_industries)



# PLAYGROUND
industry_equilibrium(c_curvature, c_shifter, z_draws[0,:], prod_grid, probs, W, gamma, theta, "Bertrand", 0.1) 

share, hhi, markupj, Pj, Zj, cost = industry_equilibrium(c_curvature, c_shifter, z_draw, prod_grid, probs, W, gamma, theta, "Bertrand", 0.1)


# QMM2, Josh's problem set
import numpy as np
from scipy.stats import pareto
from scipy import optimize
import time
import scipy as sp
from prodgrid import gen_prod_grid
from industry_eq import industry_equilibrium, price_aggregator, get_entry_costs
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"

import pandas as pd

import ipdb

def agg_variables(ratio, P, P_tildej, Z_js, markups, marginal_entrants, theta, c_shifter, c_curvature):
	''' Gives aggregated variables from industry variables
	Note, we use means to aggregate as there are infinitely tiny industries (integral 0 to 1)
	ratio is the W**(-theta) * Y guess
	'''
	W = find_wage(P_tildej, theta)
	Y = W**theta * ratio
	#Y = W**theta # TODO: slide 43, from labor demand of each firm? Maybe not in dynamic case. 
	Pj =  P_tildej * W 
	Yj = (Pj/P)**(-theta) * Y # Slide 19, L1
	costj = Yj / np.array(Z_js)
	# Slide 46, L1 + using relationship of industry markup to find expression for aggregate markup.
	Markup = ((P/W)**(1 - theta)) / np.mean(markups ** (-theta) * Z_js**(theta - 1))
	Z = (np.mean((markups/Markup)**(-theta) * Z_js**(theta-1)))**(1/(theta - 1))
	Lprod = Y/Z
	#L = Y / (Markup * W) # Slide 51 L1 Same as above for P = 1

	entry_labor = 0
	for jj in range(Z_js.size):
		entrant_vec = np.arange(marginal_entrants[jj] - 1) # Function takes incumbent as argument
		entry_labor += np.sum(get_entry_costs(W, c_shifter, entrant_vec, c_curvature))/W
	
	L_total = Lprod + entry_labor / Z_js.size
	avg_firms = np.mean(marginal_entrants)
	return(Lprod, L_total, W, Y, Z, Markup, costj, avg_firms)

def agg_equilibrium(c_curvature, c_shifter, z_draws, n_industries, prod_grid, probs, W, gamma, theta, competition, learning_rate, path):
	''' Computes an aggregate equilibrium for the entry game economy. Returns cost weighted variables by each percentile

	Conmputes each industry as if W = 1. This is then updated according to slide 42, L1, which gives us true industry prices. 


	'''
	# Industry variables, arrays with one value per industry
	shares, hhis, markups, prices_j, Z_js, marginal_entrants = [], [], [], [], [], []
	
	for industry in range(n_industries):
		print(industry)
		z_draw = z_draws[industry, :]
		share, hhi, markupj, P_tildej, Zj, marginal_entrant = industry_equilibrium(c_curvature, c_shifter, z_draw, prod_grid, probs, W, gamma, theta, competition, learning_rate) 

		shares.append(share), 
		hhis.append(hhi)
		markups.append(markupj)
		prices_j.append(P_tildej)
		Z_js.append(Zj)
		marginal_entrants.append(marginal_entrant)

	# Find aggregate price by aggregating. I.e. not equal to one here. 
	P = (np.mean(np.array(prices_j)**(1-theta)))**(1/(1-theta))

	Lprod, L, W, Y, Z, Markup, costj, avg_firms = agg_variables(1, P, np.array(prices_j), np.array(Z_js), np.array(markups), np.array(marginal_entrants), theta, c_shifter, c_curvature)

	df = pd.DataFrame(list(zip(shares, hhis, markups, prices_j, Z_js)), columns =['shares', 'hhis', 'markups', 'Ptildej', 'Zj'])
	df["cost"] = costj
	df["Pj"] = df["Ptildej"] * W # Slide 42, L1.
	df["pct"] = df.shares.rank(pct = True) * 100
	df["pct"] = df["pct"].apply(lambda x: int(x))
	df["j"] = df.index

	weights = df[["cost", "pct", "j"]].copy().set_index(["pct", "j"])
	group_weight = weights.groupby(level = 0).sum()
	weights = weights.join(group_weight, rsuffix = "_group")
	weights = weights.div(weights["cost_group"], axis = 0).drop(["cost_group"], axis = 1).rename(columns = {"cost" : "weight"})

	df = df.set_index(["pct", "j"])
	df = df.join(weights)
	# Multiply columns by cost variable used to weigh observations
	df = df.mul(df["weight"], axis=0)
	df = df.groupby(level = 0).sum()
	
	plot_weighted_values(df, "shares", path)
	plot_weighted_values(df, "hhis", path)
	plot_weighted_values(df, "markups", path)
	plot_weighted_values(df, "Pj", path)
		
	return(Lprod, L, W, Y, Z, Markup)

def plot_weighted_values(df, outcome_var, path):
	# TODO: color by max prodcutivty
	fig = px.scatter(x = df.index, y = df[outcome_var],
		labels=dict(x="Largest firm percentile", y = outcome_var), title = "Weights by industry cost")
	fig.write_image(path + "_" + outcome_var + ".png")
	return

def find_wage(Ptildej, theta):
	''' Find economy wage
	Slide 42, lecture 1
	Assumes aggregate good has a normalized price, P = 1
	'''
	
	W_inv = (np.mean(Ptildej**(1-theta)))**(1/(1-theta))
	
	W = W_inv**(-1)
	return(W)

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

if __name__ == 'main':
	alpha = 6
	gamma = 10.5
	theta = 1.24
	prod_grid, probs = gen_prod_grid(alpha, gamma)

	n_industries = 1000
	z_draws = np.random.choice(prod_grid, size = (n_industries, 5000), p = probs)
	# Note on W: suppose W**(-theta)Y = 1. Calcualte equilibrium as if W and Y are 1.
	# Compute final good price, then use PW = 1, set Y st W**(-theta)Y = 1. 
	# Finally, set L equal to aggregate labor used across industries by firms for proeuction and entry costs. 
	# To 
	W = 1
	# PART A
	c_curvature = 0
	c_shifter = 2e-4

	agg_equilibrium(c_curvature, c_shifter, z_draws, n_industries, prod_grid, probs, W, gamma, theta, "Bertrand", 1, "figures/parta")

	# PART B
	c_curvature = 1
	c_shifter = 2e-7

	Lprod, L_supply, W, Y, Z, Markup =agg_equilibrium(c_curvature, c_shifter, z_draws, n_industries, prod_grid, probs, W, gamma, theta, "Bertrand", 1, "figures/partb")

	f = open("agg_values_partb.txt", "w")
	f.write("L_prod:" + str(Lprod) + "\n L_supply:" + str(L_supply) + "\n Z" + str(Z) + "\n Markup: " + str(Markup))
	f.close()

	f = open("total_labor.txt", "w")
	f.write(str(L_supply))
	f.close()

	
	# NOTE: When I don't force P = 1, Lprod and L differ:
#Lprod                                                                                   
# Out[12]: 0.27631736586038536 
#In [13]: L                                                                                       
#Out[13]: 0.8665185419497495  


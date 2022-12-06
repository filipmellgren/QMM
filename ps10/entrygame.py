# QMM2, Josh's problem set
import numpy as np
from scipy.stats import pareto
from scipy import optimize
import time
import scipy as sp
from prodgrid import gen_prod_grid
from industry_eq import industry_equilibrium, price_aggregator
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"

import pandas as pd

import ipdb
def agg_equilibrium(c_curvature, c_shifter, z_draws, n_industries, prod_grid, probs, W, gamma, theta, competition, learning_rate, path):
	''' Computes an aggregate equilibrium for the entry game economy. Returns cost weighted variables by each percentile

	Conmputes each industry as if W = 1. This is then updated according to slide 42, L1, which gives us true industry prices. 

	'''
	# Industry variables, arrays with one value per industry
	shares, hhis, markups, prices_j, Z_js = [], [], [], [], []
	
	for industry in range(n_industries):
		print(industry)
		z_draw = z_draws[industry, :]
		share, hhi, markupj, P_tildej, Zj = industry_equilibrium(c_curvature, c_shifter, z_draw, prod_grid, probs, W, gamma, theta, competition, learning_rate) 

		shares.append(share), 
		hhis.append(hhi)
		markups.append(markupj)
		prices_j.append(P_tildej)
		Z_js.append(Zj)
	
	W = find_wage(np.array(prices_j), theta)
	P = 1
	Y = W**theta
	Pj =  np.array(prices_j) * W 
	Yj = (Pj/P)**(-theta) * Y # Slide 19, L1
	costj = Yj / np.array(Z_js)
	# Slide 46, L1 + using relationship of industry markup to find expression for aggregate markup.
	Markup = ((P/W)**(1 - theta)) / np.sum(np.array(markups) ** (-theta) * np.array(Z_js)**(theta - 1))
	Z = (np.mean((np.array(markups)/Markup)**(-theta) * np.array(Z_js)**(theta-1)))**(1/(theta - 1))
	Lprod = Y/Z
	L = Y / (Markup * W) # Slide 51 L1
	# TODO L_bar
	#L_entry = entry_costs/W Can figure these out given n and epsilon 
	#Lbar = L_prod + L_entry ?
	#L_bar = (Y - entry_costs)/Z

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
		
	return

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
	W_inv = price_aggregator(Ptildej, theta)
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

agg_equilibrium(c_curvature, c_shifter, z_draws, n_industries, prod_grid, probs, W, gamma, theta, "Bertrand", 1, "figures/partb")



# PLAYGROUND

0.94 * prod_grid@probs

partb = agg_equilibrium(1, 2e-7, z_draws, n_industries)
industry_equilibrium(c_curvature, c_shifter, z_draws[0,:], prod_grid, probs, W, gamma, theta, "Bertrand", 0.1) 

share, hhi, markupj, Pj, Zj, cost = industry_equilibrium(c_curvature, c_shifter, z_draw, prod_grid, probs, W, gamma, theta, "Bertrand", 0.1)


# Problem set 5, Bequests
import	scipy as sp
import numpy as np
import sys
sys.path.append('../')
from src.lifecycle.lc_aux_funcs import create_shock_panel, lc_policies, simulate_hh, get_bequest_transition_matrix, create_income_states, create_hh_bequests
from src.lifecycle.lc_calibrate import calibrate_life_cyc
import pandas as pd
import plotly.express as px
import time
import ipdb
#from src.reward_funcs import calc_consumption
# TODO: I multiply with bequests inside sim_years(...) where bequests should simply be additive. 

def equilibrium_distance(guess, shocks, params, phi1):
	''' Want to find a fixed point, that's our equilibrium

	'''
	# TODO:
	tmp, bequest_shock_probs = get_bequest_transition_matrix(guess, params)
	
	
	# One income state vector for each year
	income_states = create_income_states(params["income_shock"], params["determ_inc"], params)

	hh_bequests = create_hh_bequests(bequest_shock_probs, guess, params)

	policies = lc_policies(params, income_states, params["transition_matrix"], bequest_shock_probs, phi1, guess)	

	hh_panel_y, hh_panel_a, hh_panel_s, hh_panel_c = simulate_hh(params["n_hh"], income_states, shocks, policies, hh_bequests, params)
	
	death_ix = pd.DataFrame(hh_panel_s).apply(pd.Series.last_valid_index) 
	bequests = hh_panel_s[np.nan_to_num(death_ix,68).astype(int), death_ix.index] # TODO: why are there nan here?

	pb = bequests[bequests == 0].shape[0] / bequests.shape[0]
	mu = np.mean(bequests[bequests != 0])
	sigma2 = np.var(bequests[bequests != 0])

	return(np.sum((guess - np.array([pb, mu, sigma2]))**2), hh_panel_y, hh_panel_a, hh_panel_s, hh_panel_c, bequests, death_ix)

def plot_series(vector, label_dict):
	df = pd.DataFrame(np.nanmean(vector, axis = 1), columns = ["yvar"])
	df["age"] = df.index #+ min_age
	fig = px.line(df, x="age", y="yvar", template = 'plotly_white', labels = label_dict)
	return(fig)

def gini(x):
	total = 0
	for i, xi in enumerate(x[:-1], 1):
		total += np.nansum(np.abs(xi - x[i:]))
	return total / (len(x)**2 * np.nanmean(x))

def plot_bequests(phi1):
	params = calibrate_life_cyc(0.97, phi1)
	guess0 = np.array([ 0.4,  3., 10])
	economy_shocks = create_shock_panel(params, params["transition_matrix"], params["min_age"], params["max_age"])

	outcome = sp.optimize.minimize(fun = lambda x: equilibrium_distance(x, economy_shocks, params, phi1)[0], x0 = guess0, method = "Nelder-Mead")
	equilibrium_vector = outcome["x"]
	error = outcome["fun"]

	np.savetxt(f'figures/eq_vec_phi1_{phi1}.csv', equilibrium_vector, delimiter=',')
	np.savetxt(f'figures/nm_error_{phi1}.csv', error, delimiter=',')

	diff, hh_panel_y, hh_panel_a, hh_panel_s, hh_panel_c, bequests, death_ix = equilibrium_distance(equilibrium_vector, economy_shocks, params, phi1)

	fig = plot_series(hh_panel_c, dict(age = "Age", yvar="# Goods"))
	fig.update_layout(title = "Average consumption")
	fig.write_image(f'figures/consumption_{phi1}.png')

	fig = plot_series(hh_panel_y, dict(age = "Age", yvar="# Goods"))
	fig.update_layout(title = "Average income")
	fig.write_image(f'figures/income_{phi1}.png')

	beq_df = pd.DataFrame(np.array([bequests, death_ix ]).T, columns = ["bequests", "age"]).groupby("age").sum().reset_index()
	beq_df.bequests = beq_df.bequests/params["n_hh"]
	fig = px.line(beq_df, x="age", y="bequests", template = 'plotly_white')
	fig.update_layout(title = "Average bequests left")
	fig.write_image(f'figures/bequests_{phi1}.png')

	asset_df = pd.DataFrame(np.nanmean(hh_panel_a, axis = 1), columns = ["Assets"])
	asset_df["age"] = asset_df.index #+ min_age

	df = asset_df.merge(beq_df, on='age', how='left')
	df["Net_assets"] = df["Assets"] - df["bequests"].fillna(0)

	fig = px.line(df, x="age", y="Net_assets", template = 'plotly_white')
	fig.update_layout(title = "Average Assets")
	fig.write_image(f'figures/avg_assets_{phi1}.png')

	ginilist = []
	for year in range(hh_panel_a.shape[0]):
		g = gini(hh_panel_a[year,:])
		ginilist.append(g)

	ginidf = pd.DataFrame(ginilist, columns = ["gini"])
	ginidf["year"] = ginidf.index
	fig = px.line(ginidf, x = "year", y = "gini", template = 'plotly_white')
	fig.update_layout(title = "Gini")
	fig.write_image(f'figures/gini_{phi1}.png')

	# TODO make relative to total
	pct99 = np.nanpercentile(hh_panel_a, 99, axis = 1)
	pct95 = np.nanpercentile(hh_panel_a, 95, axis = 1)
	pct80 = np.nanpercentile(hh_panel_a, 80, axis = 1)

	df = pd.DataFrame(np.asarray([pct99, pct95, pct80]).T, columns = ["p99", "p95", "p80"])
	df["year"] = df.index
	df = pd.melt(df, id_vars = ["year"], value_vars = ["p99", "p95", "p80"])
	fig = px.line(df, x = "year", y = "value", color = "variable", template = 'plotly_white')
	fig.update_layout(title = "Inequality")
	fig.write_image(f'figures/inequality_{phi1}.png')
	return	

plot_bequests(phi1 = 0)

plot_bequests(-10)


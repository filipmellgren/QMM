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
	transition_matrix, bequest_shock_probs = get_bequest_transition_matrix(guess, params)
	
	# One income state vector for each year
	income_states = create_income_states(params["income_shock"], params["determ_inc"], params["bequest_grid"], params)

	hh_bequests = create_hh_bequests(bequest_shock_probs, guess, params)

	policies = lc_policies(params, income_states, transition_matrix, phi1)	

	hh_panel_y, hh_panel_a, hh_panel_s, hh_panel_c = simulate_hh(params["n_hh"], income_states, shocks, policies, hh_bequests, params)

	death_ix = pd.DataFrame(hh_panel_s).apply(pd.Series.last_valid_index) 
	bequests = hh_panel_s[death_ix, death_ix.index]

	pb = bequests[bequests == 0].shape[0] / bequests.shape[0]
	mu = np.mean(bequests[bequests != 0])
	sigma2 = np.var(bequests[bequests != 0])

	return(np.sum((guess - np.array([pb, mu, sigma2]))**2), hh_panel_y, hh_panel_a, hh_panel_s, hh_panel_c, bequests, death_ix)

if name == "__main__":

	params = calibrate_life_cyc(0.97, phi1 = 0)
	
	economy_shocks = create_shock_panel(params, params["transition_matrix"], params["min_age"], params["max_age"])

	guess0 = np.array([0.548, 2.23, 9.048])
	
	phi1  = 0

	# TODO return economy variables for plotting. Turn itno lambda funciton. 

	outcome = sp.optimize.minimize(fun = lambda x: equilibrium_distance(x, economy_shocks, params, phi1)[0], x0 = guess0, method = "Nelder-Mead")

	equilibrium_vector = outcome["x"]


	diff, hh_panel_y, hh_panel_a, hh_panel_s, hh_panel_c, bequests, death_ix = equilibrium_distance(guess0, economy_shocks, params, phi1)


	def plot_series(vector, label_dict):
		df = pd.DataFrame(np.nanmean(vector, axis = 1), columns = ["yvar"])
		df["age"] = df.index #+ min_age
		fig = px.line(df, x="age", y="yvar", template = 'plotly_white', labels = label_dict)
		return(fig)

fig = plot_series(hh_panel_c, dict(age = "Age", yvar="# Goods"))
fig.update_layout(title = "Average consumption")


economy_shocks[economy_shocks>100]
shocks = economy_shocks

income_states.shape[1]

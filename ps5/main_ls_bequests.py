# Problem set 5, Bequests
import	scipy as sp
import numpy as np
import sys
sys.path.append('../')
from src.lifecycle.lc_aux_funcs import create_shock_panel, lc_policies, simulate_hh
from src.lifecycle.lc_calibrate import calibrate_life_cyc
import pandas as pd
import plotly.express as px
import time
#from src.reward_funcs import calc_consumption

def equilibrium_distance(guess, shocks, bequest_grid):
	''' Want to find a fixed point, that's our equilibrium

	'''
	guess = np.array([0.5, 1, 2])

	bequest_grid = params["bequest_grid"]
	bequest_grid_mid_points = bequest_grid + np.mean(np.diff(bequest_grid))/2 # Add half step size
	bequest_shock_cdf= sp.stats.lognorm.cdf(bequest_grid_mid_points, s = guess[2], loc = guess[1])
	bequest_shock_probs = np.diff(np.append(0, bequest_shock_cdf)) # TODO: almost 1 what to do?
	bequest_shock_probs = bequest_shock_probs / np.sum(bequest_shock_probs) # Scale to 1
	bequest_trans_mat = np.tile(bequest_shock_probs, (bequest_shock_probs.shape[0], 1))
	transition_matrix = np.kron(params["transition_matrix"], bequest_trans_mat)
	
	policies = lc_policies(params, transition_matrix)

	hh_panel_y, hh_panel_a, hh_panel_s, hh_panel_c = simulate_hh(100, shocks, policies, params)

	death_ix = pd.DataFrame(hh_panel_s).apply(pd.Series.last_valid_index) # TODO: does it matter when bequests are left?
	bequests = hh_panel_s[death_ix, death_ix.index]

	pb = bequests[bequests == 0].shape[0] / bequests.shape[0]
	mu = np.mean(bequests[bequests != 0])
	sigma2 = np.var(bequests[bequests != 0])

	return((guess - np.array([pb, mu, sigma2]))**2)

if name == "__main__":

	params = calibrate_life_cyc(0.97)
	
	economy_shocks = create_shock_panel(params, params["min_age"], params["max_age"], params["income_shock"]) # TODO: add bequest_shocks. TODO: maybe only add bequessts on top as these will vary and the rest won't.

	guess0 = np.array([0,1,0.5])

	sp.minimize(fun = equilibrium_distance, x0 = guess0, method = "Nelder-Mead", phi1 = 0, shocks = economy_shocks, bequest_grid = bequest_grid)

	def plot_series(vector, label_dict):
		df = pd.DataFrame(np.nanmean(vector, axis = 1), columns = ["yvar"])
		df["age"] = df.index #+ min_age
		fig = px.line(df, x="age", y="yvar", template = 'plotly_white', labels = label_dict)
		return(fig)



params = calibrate_life_cyc(0.97)	
economy_shocks = create_shock_panel(params, params["min_age"], params["max_age"], params["income_shock"]) 
shocks = economy_shocks
policies = lc_policies(params, params["transition_matrix"])

start = time.time()
hh_panel_y, hh_panel_a, hh_panel_s, hh_panel_c = simulate_hh(1000, shocks, policies, params)
end = time.time()
print(end - start)

fig = plot_series(hh_panel_c, dict(age = "Age", yvar="# Goods"))
fig.update_layout(title = "Average consumption")
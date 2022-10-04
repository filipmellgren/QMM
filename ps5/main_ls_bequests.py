# Problem set 5, Bequests
import	scipy as sp
import numpy as np
import sys
sys.path.append('../')
from src.lifecycle.lc_aux_funcs import create_shock_panel, lc_policies, simulate_hh, get_bequest_transition_matrix
from src.lifecycle.lc_calibrate import calibrate_life_cyc
import pandas as pd
import plotly.express as px
import time
#from src.reward_funcs import calc_consumption
# TODO: alter utility function
# TODO: implement estate tax

def equilibrium_distance(guess, shocks, params):
	''' Want to find a fixed point, that's our equilibrium

	'''
	transition_matrix, bequest_shock_probs = get_bequest_transition_matrix(guess, params)
	
	policies = lc_policies(params, transition_matrix)

	# Add bequest shocks
	# TODO: turn into function
	for hh in range(shocks.shape[1]):
		y = 40 # When everyone is bequested in this economy
		bequest_size = np.argmin(np.abs(np.random.uniform() - bequest_shock_probs))
		bequest_size = min(bequest_size, np.exp(guess[1] + 3 * guess[2]))
		shocks[y, hh] = shocks[y, hh] + (np.random.uniform() > guess[0]) * params["bequest_grid"][bequest_size]


	income_states = np.kron(params["income_shock"], np.ones(params["bequest_grid"].shape[0]))
	income_states = income_states + np.tile(params["bequest_grid"],params["income_shock"].shape[0])

	hh_panel_y, hh_panel_a, hh_panel_s, hh_panel_c = simulate_hh(shocks.shape[1], income_states, shocks, policies, params)

	death_ix = pd.DataFrame(hh_panel_s).apply(pd.Series.last_valid_index) # TODO: does it matter when bequests are left?
	bequests = hh_panel_s[death_ix, death_ix.index]

	pb = bequests[bequests == 0].shape[0] / bequests.shape[0]
	mu = np.mean(bequests[bequests != 0])
	sigma2 = np.var(bequests[bequests != 0])

	return(np.sum((guess - np.array([pb, mu, sigma2]))**2))

if name == "__main__":

	params = calibrate_life_cyc(0.97)
	
	economy_shocks = create_shock_panel(params, params["transition_matrix"], params["min_age"], params["max_age"], params["income_shock"])

	guess0 = np.array([0.552, 1.23, 2.911])

	# TODO return economy variables for plotting. Turn itno lambda funciton. 

	outcome = sp.optimize.minimize(fun = equilibrium_distance, x0 = guess0, method = "Nelder-Mead", args=(economy_shocks, params))

	equilibrium_vector = outcome["x"]

def plot_series(vector, label_dict):
		df = pd.DataFrame(np.nanmean(vector, axis = 1), columns = ["yvar"])
		df["age"] = df.index #+ min_age
		fig = px.line(df, x="age", y="yvar", template = 'plotly_white', labels = label_dict)
		return(fig)

params = calibrate_life_cyc(0.97)	
economy_shocks = create_shock_panel(params, params["transition_matrix"], params["min_age"], params["max_age"], params["income_shock"]) 
shocks = economy_shocks


guess = np.array([0.5, 1, 2])

get_bequest_transition_matrix




income_states = np.kron(params["income_shock"], params["bequest_grid"]).flatten()

shocks[:,0] += bequest_grid * bequest_shock_probs * 
policies = lc_policies(params, np.kron(params["transition_matrix"], bequest_trans_mat))

start = time.time()
hh_panel_y, hh_panel_a, hh_panel_s, hh_panel_c = simulate_hh(1000, shocks, policies, params)
end = time.time()
print(end - start)

fig = plot_series(hh_panel_c, dict(age = "Age", yvar="# Goods"))
fig.update_layout(title = "Average consumption")
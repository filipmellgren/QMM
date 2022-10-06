import numpy as np
import pandas as pd
from src.tauchenhussey import tauchenhussey, floden_basesigma 
import ipdb

def calibrate_life_cyc(income_permamence, phi1):
	'''
	income_permamnence is the AR1 term of the icnome process
	'''
	params = {
	"disc_fact": 0.95,
	"rate": 0.04,
	"N_work": 40,
	"N_ret": 30,
	"risk_aver": 1.5,
	"pensions_rr": 0.6,
	"perm_inc_dim": 5,
	"trans_inc_dim": 2,
	"action_size": 1000,
	"n_hh": 2000,
	"min_asset": 0,
	"max_asset": 150,
	"bequest_states_n": 20,
	"estate_tax": 0.15,
	"phi2": 10
	}


	params["surv_prob"] = np.concatenate((np.full((params["N_work"]), 1), np.linspace(1, 0.92, params["N_ret"])), axis = None)
	sigma = np.sqrt(0.015)
	baseSigma = floden_basesigma(sigma, income_permamence)

	perm_inc, perm_transition_matrix = tauchenhussey(
		params["perm_inc_dim"], # number of nodes for Z
		0, # unconditional mean of process
		income_permamence, # rho
		sigma, # std. dev. of epsilons
		baseSigma) # std. dev. used to calculate Gaussian 

	p_u = 0.01
	temp_transition_matrix = np.array([[1 - p_u, p_u], [1 - p_u, p_u]])
	params["determ_inc"] = pd.read_csv("data/lc_profile.csv")
	# Create transition matrix for each time period
	params["transition_matrix"] = np.kron(temp_transition_matrix, perm_transition_matrix)
	params["income_shock"] = np.kron(np.array([1, 0.4]), np.exp(perm_inc)).flatten()

	params["bequest_grid"] = np.linspace(params["min_asset"],params["max_asset"], params["bequest_states_n"])


	params["action_states"] = np.logspace(
		start = params["min_asset"], 
		stop = np.log10(params["max_asset"] - params["min_asset"]+1), 
		num = params['action_size']) + params["min_asset"] -1
	
	params["exog_grid"] = np.repeat(np.expand_dims(params["action_states"], axis = 1), params["income_shock"].shape[0], axis = 1)

	params["min_age"] = params["determ_inc"].age.min()
	params["max_work_age"] = params["determ_inc"].age.max()
	params["max_age"] =  params["max_work_age"] + params["N_ret"]

	# BELOW TODO
	G_ret = params["determ_inc"].income.iloc[-1] * params["pensions_rr"]
	params["terminal_income_states"] = np.ones(params["income_shock"].shape[0]) * G_ret

	if phi1 == 0:
		params["terminal_policy"] = np.zeros((params["action_size"], params["income_shock"].shape[0])) # Last policy is to save nothing. 1000 by 10
		return(params)
		
	# Solve for terminal policy
	# icnome shock 10 by 1
	k = ((1-params["estate_tax"]) * (1 - params["risk_aver"]) * phi1/params["phi2"])**(-1/params["risk_aver"]) # scalar

	terminal_policy = (-k + params["exog_grid"] * (1 + params["rate"]) + G_ret * np.ones(params["exog_grid"].shape))/(1 + (1-params["estate_tax"])/(params["phi2"]) * k)
	params["terminal_policy"] = terminal_policy

	return(params)
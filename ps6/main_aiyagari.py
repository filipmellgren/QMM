# Ayiagari model
import numpy as np
import scipy as sp
import sys
sys.path.append('../')
from src.tauchenhussey import tauchenhussey, floden_basesigma 
from src.vfi_funcs import egm
from src.distribution_funcs import kieran_finder

import ipdb

def market_clearing(r_in):
	alpha = 0.3
	sigma = np.sqrt(0.04)
	rho = 0.94
	mu = 0
	inc_size = 5

	baseSigma = floden_basesigma(sigma, rho)
	log_income_states, transition_matrix = tauchenhussey(inc_size, mu,rho,sigma, baseSigma)
	income_states = np.exp(log_income_states[0,:])

	income_states = income_states / np.mean(income_states)
	wage = (1-alpha)
	income_states = wage * income_states

	min_assets = -1
	max_assets = 150

	asset_grid = np.logspace(
		start = min_assets, 
		stop = np.log10(max_assets - min_assets+1), 
		num = 1501) + min_assets -1

	params = {
	"disc_fact": 0.96,
	"risk_aver": 1.5,
	"income_states": income_states
	}

	distr_guess = np.zeros((asset_grid.shape[0], inc_size)) + 0.1

	policy = egm(transition_matrix, asset_grid, r_in, params, tol = 1e-6)
	ergodic_distribution = kieran_finder(distr_guess, policy, asset_grid, transition_matrix, inc_size, asset_grid.shape[0], tol = 1e-3)
	K_hh = np.sum(ergodic_distribution * policy)
	
	delta = 0.1
	K_firm = ((r_in + delta)**(1-alpha))/(alpha)
	diff = K_hh-K_firm
	return(diff)

r_star = sp.optimize.bisect(market_clearing, 0, 0.1) # 0.0418




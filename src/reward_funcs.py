import numpy as np
import ipdb

def find_asset_grid(params, borrow_constr):
	asset_grid = np.logspace(start = 0, stop = np.log10(params['asset_max'] - borrow_constr +1), num = params['action_grid_size']) + borrow_constr -1
	return(asset_grid)

def get_util(consumption, params):
	'''
	Calculate agent dirtect utility. 
	TODO: implement option to specify functional form
	'''
	utility = np.zeros(consumption.shape)
	with np.errstate(invalid='ignore'):
		utility = (np.power(consumption, 1-params["risk_aver"])-1)/(1-params["risk_aver"])
	utility[consumption<0] = -np.inf 

	assert utility[consumption.shape[0]-1, consumption.shape[1]-1, 0] == np.max(utility), "Expected max utility is not max."
	return(utility)

def calc_consumption(params, rate, borrow_constr):
	''' Calculates consumption for each state action pair

	Parameters
	----------
	rate : a scalar interest rate  in the economy.
	borrow_constr : a np.array where borrow_constr[0] = today's borrow constraint, and 
		borrow_constr[1] = tomorrow's borrow constraint.
	Returns
	-------
	consumption
	  a np array if 3 dimensions with the total goods consumed in each state-action combination
	'''
	income_states = params["income_states"]
	try:
		asset_states = find_asset_grid(params, borrow_constr[0])
		action_set = find_asset_grid(params, borrow_constr[1])
	except TypeError as e:
		print("Need to make the borrow constraint a vector with todays Br.C and tomorrow's Br.C.")
		raise e
	
	try:
		income, asset, saving = np.meshgrid(income_states, asset_states, action_set, sparse = True, indexing='ij')
	except SystemError:
		print("SystemError, what the @*!$€ is wrong?")
		ipdb.set_trace()
	
	consumption = income + asset * (1+rate) - saving

	# Check, all else equal, that consumption increases with higher values
	arbitrary_index_ass = np.random.randint(0, high=asset_states.shape[0]-1)
	arbitrary_index_act = np.random.randint(0, high=action_set.shape[0]-1)

	income_dim = consumption[:,arbitrary_index_ass,arbitrary_index_act]
	asset_dim = consumption[0,:,arbitrary_index_act]
	savings_dim = consumption[0,arbitrary_index_ass,:]

	if not np.all(income_dim[1:] > income_dim[:-1]):
		ipdb.set_trace()
	assert np.all(income_dim[1:] > income_dim[:-1]), "Consumption not increasing in income"
	
	if not np.all(asset_dim[1:] > asset_dim[:-1]):
		ipdb.set_trace()
	assert np.all(asset_dim[1:] > asset_dim[:-1]), "Consumption not increasing in assets"
	assert np.all(savings_dim[1:] <= savings_dim[:-1]), "Consumption not decreasing with savings"
	assert consumption.shape == (income_states.shape[0], asset_states.shape[0], action_set.shape[0])
	assert consumption[0,0,action_set.shape[0]-1] == np.min(consumption), "Expected min consumption is not actual min consumption"
	return(consumption)

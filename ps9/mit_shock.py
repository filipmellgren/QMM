import numpy as np
# Solve for nonlinear steqady state. DOne!


# Solve for transition path to small shock (e.g. TFP) imposing market clearing
def solve_trans_path(ss, T, distr0, policy_ss, K_guess):

	#  Guess path for capital and labor
		# K can be a linear decay
	K0 = ss.K 
	diff = 100
	tol = 1e-2
	weight = 1e-6

	# TODO: look at capital to output ratio. Scale by shock size for guess of capital in the shock period, then decay this
	#K_guess = np.linspace(38.4, K, num = T+1)
	#K_guess[0] = K
	
	shock = 0.01
	tfp =  1 + shock * 0.95**np.linspace(0,T, T)
	tfp = np.insert(tfp, 0, 1)
	tfp[-1] = 1
	
	tfp[0] = 1
	n_iters = 0
	#distr_guess = [np.full(ss.state_grid.shape, 1)/len(ss.state_grid)]

	K_HH_list = []

	while diff > tol and n_iters < 200:
		n_iters += 1
		# Solve the value function/household policy backwards
		
		policy = backward_iterate_policy(tfp, K_guess, policy_ss, T, ss)
		
		# From SS, solve forward using policy functions and idiosyncratic Markov process
		distr = forward_iterate_distribution(policy, distr0, T, ss)
		
		
		# Calculate capital supplied at each point in time
		flat_pols = []
		for pol in policy:
			flat_pols.append(pol.flatten())
		K_HH = np.sum(np.asarray(flat_pols) * np.asarray(distr), axis = 1)
		K_HH = K_HH[0:T] # HH savings in period t # Should correspond to assets in t+1

		K_HH_list.append(K_HH)

		# Check max difference # TODO: K_hh_t is distr * savings whereas k_guess is total assets. Need to compare to values in the next period
		diff_array = K_HH - K_guess[1:T+1]
		diff = np.max(diff_array)**2
		diff_ix = np.argmax(np.abs(diff_array))
		
		# Update guess for K
		K_guess = (1- weight) * K_guess[1:T+1] + (weight) * K_HH
		K_guess = np.insert(K_guess, 0, K0)
		print((diff_array[diff_ix], weight))
		weight = np.maximum(weight*0.99, 1e-7)

	return(K_guess, K_HH_list, tfp, T)


def backward_iterate_policy(tfp, K_guess, policy_ss, T, ss):
	''' Iterate backwards to get policies in transition.
	ss is an instance of Market
	'''
	policy_list = [policy_ss]
	for time in reversed(range(T)):
		tfp_t = tfp[time]
		K_t = K_guess[time]
		ss.set_tfp(tfp_t, K_t)
		#ss.set_capital(K_t) # OBS: this updates the market object's rate (so it is no longer steady state)
		policy_prev = policy_list[-1]
		policy_t = egm_update(policy_prev, ss.P, ss.r, ss.wage, ss.tax, ss.L_tilde, ss.mu, ss.gamma, ss.beta, ss.delta, ss.state_grid, ss.asset_states) # Can use egm_update, do not iterate on it
		policy_list.append(policy_t)
	
	policy_list.reverse()
	policy = np.asarray(policy_list)
	
	return(policy)


def forward_iterate_distribution(policy, distr0, T, ss):
	''' Forward iterate to find ergodic distributions
	This function is kind of slow compared to the backward iteration.
	distr_guess : a guess for the distribution. A good guess is the distribution from a previous iteration. Is a list of length T.
	'''
	# TODO: what distribution corespnds to what olicty
	distr_list = [distr0]
	for time in range(T):
		policy_t = policy[time]
		#policy_t_ix = value_array_to_index(policy_t.flatten(), ss.asset_states)
		policy_t_ix, alpha_list = policy_to_grid(policy_t.flatten(), ss.asset_states)
		P = get_transition_matrix(ss.Q, nb.typed.List(policy_t_ix), nb.typed.List(alpha_list), ss.state_grid) # TODO: is Q not changing? No, but P is, and that updates
		P = np.asarray(P)
		try:
			distr_t = get_distribution(P)
		except:
			ipdb.set_trace()
		#distr_t = get_distribution_iterate(distr_guess[time], P, ss.state_grid, tol = 1e-5)
		#mc = qe.MarkovChain(P)
		#distr_t = mc.stationary_distributions[0]
		distr_list.append(distr_t)
	distr = np.asarray(distr_list)
	return(distr)

if name == "__main__":
	ss = steady_state
	T = 300
	K_guess = np.repeat(ss.K, T+1)
	policy_ss = np.reshape(np.asarray(policy_ss), (2, 1000))
	K_sol, K_HH_evol, tfp_seq, T = solve_trans_path(ss, T, distr, policy_ss, K_guess)





	K_sol.shape

	import pandas as pd
	df = pd.DataFrame(K_HH_evol)

	df["iter"] = df.index

	df2 = pd.melt(df, id_vars = "iter", var_name = "time", value_name = "K")

	fig = px.line(df2, x="time", y="K", color = "iter")
	fig.show()





	df3 = pd.DataFrame(tfp_seq)
	df3 = df3.reset_index()
	fig = px.line(df3, x = "index", y = 0)
	fig.show()

	ss.K


	time = np.arange(len(tmp))
	df = pd.DataFrame(np.array([tmp, time]))



	tmp0 = tmp
	print((1, 2))

	tmp = [1,2,3,4]
	tmp[-1]
	tmp.append(5)

	# Use the Impulse Response Function as a numerically computed derivative

	# Treat the value of a variable at point t as the sum of responses to all past shocks

	tmp = np.linspace(0,10,10)
	np.insert(tmp, 0, 1000)




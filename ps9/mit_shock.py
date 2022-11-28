import numpy as np
from household_problem import egm_update, policy_to_grid
from distribution_funcs import get_transition_matrix, get_distribution, get_distribution_fast
import numba as nb
from numba import jit, prange
# Solve for nonlinear steqady state. DOne!
import ipdb

# Solve for transition path to small shock (e.g. TFP) imposing market clearing
def solve_trans_path(ss, T, distr0, policy_ss, tfp, K_guess):
	''' BKM algorithm for solving for transition path following shock in TFP.

	'''
	K0 = ss.K 
	loss = 100
	tol = 1e-4
	weight = 1e-1
		
	n_iters = 0

	K_HH_list = []

	while loss > tol and n_iters < 200:
		n_iters += 1

		policy, rate_path, wage_path = backward_iterate_policy(tfp, K_guess, policy_ss, T, ss)
		
		distr = forward_iterate_distribution(policy, distr0, T, ss)
		
		K_HH = K_supply(policy, distr, T)
		K_HH_list.append(K_HH) 

		# Check loss function
		loss = np.max(K_HH - K_guess[1:T+1])**2
				
		# Update capital demand guess
		K_guess = (1- weight) * K_guess[1:T+1] + (weight) * K_HH
		K_guess = np.insert(K_guess, 0, K0)

		weight = np.maximum(weight*1, 1e-5)

	return(K_guess, K_HH_list, tfp, T, rate_path, wage_path, policy, distr)

def K_supply(policy, distr, T):
	''' Find capital supplu from policy and distritbution of households
	
	Ensure distribution and policy are setup in a way such that the right mass coprresponds to the right policy.

	INPUT
	policy : is a numpy array of dimensions number of states not related to assets X number of possible asset states X T. It is flattened to work with the array. Interpret as one policy for each state in each period.
	distr : is a numpy array of dimensions total number of states X T. 

	OUTPUT
	K_HH : A numpy array of capital supplied for each t in T.

	'''
	flat_pols = []
	for pol in policy:
		flat_pols.append(pol.flatten())

	K_HH = np.sum(np.asarray(flat_pols) * np.asarray(distr), axis = 1)
	K_HH = K_HH[0:T] # HH savings in period t that should correspond to firm assets in period t+1
	return(K_HH)


def backward_iterate_policy(tfp, K_guess, policy_ss, T, ss):
	''' Iterate backwards to get policies in transition.
	ss is an instance of Market
	policy_ss : is the policy in steady state. A numpy array of income states X asset states containing values (not indices) of savings. 
	'''
	policy_list = [policy_ss]
	rate_fut = ss.r 
	wage_fut = ss.wage
	rate_path = [rate_fut]
	wage_path = [wage_fut]
	for time in reversed(range(T)):
		
		tfp_t = tfp[time] # First 'time' is T-1
		K_t = K_guess[time]
		ss.K = K_t # K is endogenous (from guess). Need to find fixed point st markets clear.
		ss.set_tfp(tfp_t, K_t) # Set A exogenously
		rate = ss.r
		wage = ss.wage
		
		policy_prev = policy_list[-1]

		policy_t = egm_update(policy_prev, ss.P, rate, rate_fut, wage, wage_fut, ss.tax, ss.L_tilde, ss.mu, ss.gamma, ss.beta, ss.delta, ss.state_grid, ss.asset_states) # Can use egm_update, do not iterate on it

		policy_list.append(policy_t)
		rate_fut = np.copy(rate)
		wage_fut = np.copy(wage)
		rate_path.append(rate_fut)
		wage_path.append(wage_fut)
	
	policy_list.reverse()
	rate_path.reverse()
	wage_path.reverse()
	policy = np.asarray(policy_list)
	
	return(policy, rate_path, wage_path)


def forward_iterate_distribution(policy, distr0, T, ss):
	''' Forward iterate to find ergodic distributions
	This function is kind of slow compared to the backward iteration.
	distr_guess : a guess for the distribution. A good guess is the distribution from a previous iteration. Is a list of length T.

	TODO: Don't need to iterate on the distribution. Instead, use following algorithm:
	* For each income state, 
	* For each k
	* we know their policy of savings. 2 by 1000 matrix, income states x asset states
	* Multply new matrix with little p transpose: P'D = 2 by 1000 distribution matrix.

	'''
	
	distr_list = [distr0]
	for time in range(T):
		policy_t = policy[time]
		#policy_t_ix = value_array_to_index(policy_t.flatten(), ss.asset_states)
		policy_t_ix, alpha_list = policy_to_grid(policy_t.flatten(), ss.asset_states)
		P = get_transition_matrix(ss.Q, nb.typed.List(policy_t_ix), nb.typed.List(alpha_list), ss.state_grid) 
		P = np.asarray(P)
		distr_t = get_distribution_fast(distr_list[-1], P, ss.state_grid, tol = 1e-12)
		#distr_t = get_distribution(P)
	
		distr_list.append(distr_t)
	distr = np.asarray(distr_list)
	return(distr)


# FAKE NEWS APPROACH
def analytical_jacobians():
	''' Derivatives of r_(s+1) and w_(s+1) wrt K_s and Z_s
	Remember: r_s+1 and w_s+1 are just marginal products wrt capital and labor
	pi * l = 1 in our example, normalized hours worked.
	'''
	pi_l = 1
	drdk = alpha * (alpha - 1) * tfp_ss * (K_ss/ (pi_l))**(alpha - 2)
	dwdk = (1-alpha) * alpha * tfp_ss * (K_ss/ (pi_l))**(alpha - 1)
	drdz = alpha * K_ss**(alpha-1) * pi_l**(1-alpha)
	dwdz = (1-alpha) * K**alpha * pi_l **(-alpha)
	return(drdk, dwdk, drdz, dwdz)

@jit(nopython=True, parallel = False)
def get_prediction_vectors(P_ss, y_ss, T):
	''' Get prediciton vectors P_u for all u with one transpose forward iteration
	
	See slide 19

	u is time until shock. So u = 0 is day of shock, u = 1 is date T - 1, so day before, etc.

	The steady state transition matrix P_ss gets raised to power u.

	P_ss is a sparse "n by n" matrix, where "n" is the total number of states, depending on both the initial individual state transition matrix (usually a small matrix), and the policies individuals make telling us how they traverse the state distribution.

	y_ss is an R^n vector. We can view this as the asset savings policy in steady state.

	Each prediction vector is an R^n object, so Pu is a T by n 

	INPUT
	P_ss : numpy array of overall transition matrix
	y_ss : numpy array of policies in steady state 2000 vector

	OUTPUT
	Pu : T by n array of prediciton vectors, T such vectors of length n.
	'''
	n = y_ss.shape[0]
	
	Pu = []
	P_poweru = P_ss
	for u in range(T):
		P_poweru = P_poweru @ P_ss
		Pu.append(P_poweru @ y_ss) # (n by n) @ (n by 1) = (n by 1)

	return(Pu)

@jit(nopython=True, parallel = False)
def fake_news_matrix(curlyP, curlyD, curlyY, T, S):
	''' Construct the fake news matrix
	See slide 18 from lecture for definition of caligraphuic letters
	INPUT
		curlyY : is a numpy array such that dY_t = Y_cal_(s-t), i.e. change in Y at t which is 				dy_t'D_ss (change in each individual's y holding distribution fixed at ss levels)
		curlyP is the matrix of prediction vectors
		curlyD : is a list of distribution arrays

	'''
	
	F = np.zeros((T,S))
	
	for t in range(T):
		for s in range(0, S, 1): 
			if t == 0:
				F[t, s] = curlyY[s] # TODO: seems go give weird resutls
				continue
			F[t, s] = np.vdot(curlyP[t-1,:].T, curlyD[s,:]) 
	return(F)

@jit(nopython=True, parallel = False)
def capital_jacobian(fake_news):
	''' Jacobian of the capital function wrt a variable depending on the fake_news matrix
	Could be r_s or w_s, for example
	No analytical solution to this derivative, why we use this approach. 
	
	See Eq (3) slide 21 for definition.
	'''
	J = np.copy(fake_news)
	T = J.shape[0]
	S = J.shape[1]
	
	for t in range(1,T, 1):
		for s in range(1, S, 1):
			J[t, s] += J[t-1, s-1]
	return(J)

def get_jacobians(T, P_ss, y_ss, distr, policy, h):
	''' Get the Jacobian matrix numerically

	curlyD[s] is the one-period-ahead distribution at each horizon s
	'''
	# STEP 1: Backward iterate for each input i and get Dcal_u^i
		# TODO: what is D1_noshock? THINK: the one period ahead distribution, assuming no shock hits the economy
		# TODO: what is actually curlyD? Maybe one period ahead distribution?
	curlyD = (distr[1:,] - distr[0]) / h
	curlyY = np.empty(T)
	for s in range(T):
		# slide 18 or 19
		curlyY[s] = np.vdot(distr[0], (policy[s] - y_ss).flatten()) / h
	
	# STEP 2: Forward iterate for each output o and get prediction/expectation vectors
	curlyP = get_prediction_vectors(P_ss, y_ss.flatten(), T) # TODO only until T-2 needed? Yes, curlyY is used otherwise
	curlyP = np.asarray(curlyP)
	
	# STEP 3: For each o, i, combine Ycal_u^(o,i) with matrix product of Pcal^o.T and Dcal^i to get fake news matrix F^(o,i)
		# curlyP is the matrix of prediction vectors
		# curlyD is the distribution matrix
	F = fake_news_matrix(curlyP, curlyD, curlyY, T, T)

	# STEP 4: Construct Jacobian from Fake news matrix.
	J = capital_jacobian(F)
	return(F, J)


#



def ir_K(H_K, H_Z, dZ):
	''' Change in K wrt z,. the impulse response

	'''
	H_K_inv = np.linalg.inv(H_K)
	dK = - H_K_inv @ H_Z @ dZ # Impulse response
	return(dK)


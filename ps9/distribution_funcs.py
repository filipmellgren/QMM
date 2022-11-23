
import numpy as np
from numba import jit, prange
import numba as nb
import ipdb

import scipy as sp
from scipy import linalg

#@jit(nopython = True)
def get_transition_matrix(Q, policy_ix_up, alpha_list, state_grid):
	''' From the overall transition matrix, return the transition matrix based on how agents choose. I.e. subset the action that is chosen, keeping all transitions, for each state

	alpha_list : list of fractions saying what amount goes to point on grid above. For borrowing constraint, alpha = 1.

	Compared to quantecons version
	mc = qe.MarkovChain(P)
	mc.stationary_distributions 
	
	'''
	P = []
	for state in range(len(state_grid)):
		transitions = Q[state, policy_ix_up[state], :] * (1-alpha_list[state]) +  Q[state, policy_ix_up[state] -1, :] * (alpha_list[state])
		P.append(transitions)
	return(P)

def get_distribution(P):
	''' Get ergodic distribution associated with unit eigenvalue
	Compared to quantecon's
	mc = qe.MarkovChain(P)
	mc.stationary_distributions 
	(Not exact, but similar)
	'''
	value, vector = sp.sparse.linalg.eigs(P.T, k=1, sigma=1) # Search for one eigenvalue close to sigma.
	assert np.isclose(value, 1)
	vector = np.abs(np.real(vector))
	vector = vector / np.sum(vector)

	return(np.squeeze(vector))

@jit(nopython=True, parallel = False)
def get_distribution_iterate(distr_guess, P, state_grid, tol = 1e-8):
	''' Find ergodic distribution from a transition matrix

	Note: it works but even with Numba is actually slower than finding the Eigen vector associated with the unit Eigen value

	Jonas pointed out that precision matters and that he used 1e-10.

	Input:
		distr_guess : numpy array of shape (len(state_grid),)

		P : transition matrix of shape (len(state_grid), len(state_grid))

		state_grid : grid of all possible states. Only its len matters.

	Returns:
		lambd : The Ergodic distribution. Presumbaly the one associated with the unit Eigenvalue.
	'''
	lambd = np.copy(distr_guess) # 2000 vector that sums to one
	
	diff = 100.0
	while diff > tol:

		diff = 0.0
		lambdap = np.full(lambd.shape, 0.0)
		for state_next in prange(len(state_grid)):
			for state in range(len(state_grid)):
				prob = P[state, state_next]
				if prob == 0:
					continue
				lambdap[state_next] += prob * lambd[state]
		
		diff = np.max(np.abs(lambdap - lambd))
		lambd = np.copy(lambdap)
		# Below assertion cannot be used in JIT mode
	#assert(abs(np.sum(distr_guess) - 1)<0.00001), print(np.sum(distr_guess))
	return(lambd)



@jit(nopython=True, parallel = True)
def get_distribution_fast(distr_guess, P, state_grid, tol = 1e-12):
	''' Find ergodic distribution from a transition matrix.
	NOTE: works with a GOOD initial guess. Uniform wont cut it.
	Jonas pointed out that precision matters and that he used 1e-10.

	This version does parallelization, matrix multiplication, and 
	most importantly, checks the max difference less frequently. 
	
	Input:
		distr_guess : numpy array of shape (len(state_grid),)

		P : transition matrix of shape (len(state_grid), len(state_grid))

		state_grid : grid of all possible states. Only its len matters.

	Returns:
		lambd : The Ergodic distribution. Presumbaly the one associated with the unit Eigenvalue.

	'''
	lambd = np.copy(distr_guess) # 2000 vector that sums to one
	diff = 100.0
	iteration = 0
	while diff > tol:
		iteration += 1
		diff = 0.0
		lambdap = np.full(lambd.shape, 0.0)
		for state_next in prange(len(state_grid)):
			probvec = P[:, state_next]
			lambdap[state_next] = probvec @ lambd
		if iteration % 500 == 0:
			# Checking the difference is an expensive operation.
			diff = np.max(np.abs(lambdap - lambd))
		lambd = np.copy(lambdap)
	return(lambd)



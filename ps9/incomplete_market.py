
import numpy as np
from numba import jit, prange
import numba as nb
import quantecon as qe
from quantecon import MarkovChain
from quantecon.markov import DiscreteDP
import ipdb
import plotly.express as px
import plotly.graph_objects as go
import time
from scipy.optimize import minimize_scalar
import scipy as sp
from scipy import linalg
from scipy.interpolate import griddata


import warnings
warnings.filterwarnings("error")

class	Market:
	''' Store data and parameterize the incomplete markets economy with aggregate risks
	'''
	def __init__(self,
		hh_panel,
		r = 0.02,
		gamma = 2, # CRRA parameter
		beta = 0.99, # Discount factor
		P = np.array([[0.47, 0.53], [0.04, 0.96]]), # Employment Markov transition matrix
		mu = 0.15, # Unemployment replacement rate
		delta = 0.025, # Capital depreciation rate
		alpha = 0.36, # Capital share
		borrow_constr = 0, # Borrowing constraint
		max_asset = 500,
		T = 1000,
		rho = 0.95,
		sigma = 0.007,
		a_size = 1000
		):
		''' Initialize the economy with given parameters
		'''
		self.r = r
		self.hh_panel = hh_panel
		self.gamma = gamma # Risk aversion
		self.beta = beta # Discount factor
		self.P = P
		self.mu = mu
		self.delta = delta # Depreciation
		self.alpha = alpha
		self.asset_states = np.logspace(start = borrow_constr, stop = np.log10(max_asset+1), num= a_size)-1
		self.sigma = sigma # TODO: what does he mean?
		self.rho = rho
		self.T = T
		self.unemployment_rate = 1 - np.mean(hh_panel) # series of unemployment
		self.L_tilde = 1 / ( 1 - self.unemployment_rate) # How much they work, given that they work. Implies L = 1
		self.L = 1
		self.A = 1 # TODO: make general
		self.capital_from_r()
		self.s_size = 2 * a_size # Number of states (TODO employed unemployed X num assets)
		self.a_size = a_size # number of actions
		self.R = np.empty((self.s_size, self.a_size))
    self.Q = np.zeros((self.s_size, self.a_size, self.s_size))
    self.state_grid = np.asarray(range(2*a_size))

    self.capital_from_r()
    self.get_wage()
    self.get_tax_rate()

    self.build_R() 
    self.build_Q()

  def capital_from_r(self):
  	self.K = (self.r/(self.alpha * self.A))**(1/(self.alpha - 1)) * self.L
  
  def r_from_capital(self):
  	self.r = self.alpha * self.A * self.K**(self.alpha - 1) * self.L**(1- self.alpha)

  def set_tfp(self, new_tfp, K):
  	''' Shock TFP, guessing new value of K
  	Updating K triggers an updating of prices, and therefore also capital
  	'''
  	self.A = new_tfp
  	self.set_capital(K)
  	


  def get_wage(self):
  	self.wage = (1 - self.alpha) * self.A * self.K**self.alpha * self.L**(-self.alpha)
  	
  def get_tax_rate(self):
  	self.tax = self.mu * self.unemployment_rate

  def set_capital(self, K):
  	''' 
  	Use this method to reset capital. This will trigger a calculation of the new interest rate
  	amd build a new reward matrix based on that.
  	'''
  	self.K = K
  	self.r_from_capital()
  	self.get_wage()
  	self.build_R()

  def set_prices(self, r):
      """
      Use this method to reset prices. Calling the method will trigger a
      re-build of R, and calculate a new implied firm capital level
      """
      self.r = r
      self.capital_from_r()
      self.get_wage()
      self.build_R()


  def build_Q(self):
      self.Q = populate_Q(self.Q, self.s_size, self.a_size, self.state_grid, self.asset_states, self.P)

  def build_R(self):
  	self.R = populate_R(
  		self.R, 
  		self.s_size,
  		self.a_size,
  		self.state_grid,
  		self.asset_states,
  		self.r,
  		self.wage,
  		self.tax,
  		self.L_tilde,
  		self.mu,
  		self.delta,
  		self.gamma)
 
	def generate_tfp_seq(self):
		path = [1]
		mean = -(self.sigma**2)/2
		for time in range(self.T):
			shock = np.random.normal(mean, self.sigma)
			path.append(np.exp(self.rho * np.log(path[time]) + shock))
		return(path)

@jit(nopython=True)
def populate_R(R, n, m, state_grid, asset_states, rate, wage, tax, labor, mu, delta, gamma):
  """
  Populate the R matrix, with R[s, a] = -np.inf for infeasible
  state-action pairs.
  """
  
  for s in range(n):
  	employed = is_employed(s, state_grid)
  	assets = assets_from_state(s, asset_states)

  	for a in range(m):
  	
  		savings = asset_states[a]
  		consumption = wage * ((1-tax) * labor * employed + mu * (1 - employed)) + (1 + rate - delta) * assets - savings
  		if consumption <= 0:
  			R[s, a] = -np.Inf
  			continue
  		R[s, a] = (consumption**(1 - gamma) - 1) / (1 - gamma)
  return(R)

@jit(nopython=True)
def populate_Q(Q, n, m, state_grid, asset_states, P):
	''' Populate the transition probability matrix
	
	Q needs to be a three-dimensional array where Q[s, a, s'] is the probability of transitioning to state s' when the current state is s and the current action is a.

	Q[s, a, s'] = P(employed, employed') * indicator whether a corresponds with assets in s'.
	'''

	for s_now in range(n):
		employed = is_employed(s_now, state_grid)

		for act in range(m):
			savings = asset_states[act]

			for s_next in range(n):
				assets_in_state = assets_from_state(s_next, asset_states)
				if assets_in_state != savings:
					Q[s_now, act, s_next] = 0 
					continue
				employed_next = is_employed(s_next, state_grid)
				Q[s_now, act, s_next] = P[int(employed), int(employed_next)]
	return(Q)

@jit(nopython=True)
def is_employed(s, state_grid):
	''' 
	state grid first has all unemployed all assets, then all employed and all assets again
	'''
	return(s > len(state_grid)/2 - 1)

@jit(nopython=True)
def assets_from_state(s, asset_grid):
	''' 
	state grid first has all unemployed all assets, then all employed and all assets again
	'''
	remainder = s % len(asset_grid)
	if remainder == 0:
		return(asset_grid[0])
	assets = asset_grid[s % len(asset_grid)]
	return(assets)

def sim_hh_panel(P, T, N_hh):
	''' Simulate a N_hh by T array of hh employment status
	Needs the quantecon package installed (not on python 3.11)
	'''
	mc = qe.MarkovChain(P, state_values = ('unemployed', 'employed'))
	hh_list = []
	for hh in range(N_hh):
		hh_trajectory = mc.simulate_indices(ts_length=T)
		hh_list.append(hh_trajectory)
	return(np.asarray(hh_list))

def solve_hh(P, rate, wage, tax, labor, mu, risk_aver, disc_factor, delta, state_grid, asset_states):
	''' Solve the HH problem
	Returns a policy vector. One savings value per possible state. 
	Comes very close to quantecon's  policy iteration values. Sometimes the policy is slightly too high relative to quantecon (never too low). Most often the same. 
	# Iterate on values. Return index. 
	'''
	
	policy_guess = np.asarray([asset_states, asset_states])
	tol = 1e-3
	diff = 100
	while diff > tol:
		diff = 0
		policy_guess_upd = egm_update(policy_guess, P, rate, wage, tax, labor, mu, risk_aver, disc_factor, delta, state_grid, asset_states)
		diff = np.max(np.abs(policy_guess_upd - policy_guess))
		policy_guess = policy_guess_upd.copy()

	policy = policy_guess.flatten() # TODO: replace with function
	policy_ix = []
	for pol in policy:
		nearest_ix = np.argmin(np.abs(pol - asset_states))
		policy_ix.append(nearest_ix)
	return(policy_ix)

def value_array_to_index(value_array, grid):
	array_ix = []
	for val in value_array:
		nearest_ix = np.argmin(np.abs(val - grid))
		array_ix.append(nearest_ix)
	return(array_ix)

@jit(nopython=True)
def mu_cons(consumption, risk_aver):
	return(consumption**(-risk_aver))

@jit(nopython=True, parallel = False)
def egm_update(policy_guess, P, rate, wage, tax, labor, mu, risk_aver, disc_factor, delta, state_grid, asset_states):
	''' Use EGM to solve the HH problem.

	'''
	# From Utility to Marginal Utility with CRRA preferences:
	#mu_cons = ((1 - risk_aver) * R - 1)**(-risk_aver / (1 - risk_aver))
	#mu_cons = (savings + assets*(1+r-delta) + income)**(-risk_aver)
	exog_grid = asset_states
	
	policy_mat = np.empty(policy_guess.shape) 

	#for state in range(len(state_grid)):
	for employed in [0,1]:
		
		income = wage * ((1-tax) * labor * employed + mu * (1 - employed))
		endog_assets = []

		for action in range(len(asset_states)):
			E_term = 0
			savings = asset_states[action]
			for state_next in range(len(state_grid)):
				# Test whether state_next is reached by action taken before proceeding:
				assets_fut = assets_from_state(state_next, asset_states)
				if assets_fut != savings:
					continue
				assets_fut_ix = np.argmin(np.abs(assets_fut - asset_states))
				employed_fut = is_employed(state_next, state_grid)

				savings_fut = policy_guess[int(employed_fut), assets_fut_ix]
				income_fut = wage * ((1-tax) * labor * employed_fut + mu * (1 - employed_fut))
				consumption_fut = income_fut + assets_fut * (1+ rate - delta)- savings_fut

				mu_cons_fut = mu_cons(consumption_fut, risk_aver)
				prob = P[int(employed), int(employed_fut)]
				E_term += prob * mu_cons_fut * (1 + rate - delta)
			
			consumption_today = (disc_factor * E_term)**(-1/risk_aver) # Rate already in expectation
			# Mapping from actions to assets_endog, given state:
			endog_assets.append((1/(1+rate-delta)) * (consumption_today + savings - income))

		endog_assets = np.asarray(endog_assets)
		policy_mat[employed,:] = np.interp(x = exog_grid, xp = endog_assets, fp = asset_states)

	return(policy_mat)

@jit(nopython = True)
def get_transition_matrix(Q, policy, state_grid):
	''' From the overall transition matrix, return the transition matrix based on how agents choose. I.e. subset the action that is chosen, keeping all transitions, for each state

	Compared to quantecons version
	mc = qe.MarkovChain(P)
	mc.stationary_distributions 
	'''

	P = []
	for state in range(len(state_grid)):
		transitions = Q[state,policy[state],:]
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

@jit(nopython=True, parallel = True)
def get_distribution_iterate(distr_guess, P, state_grid, tol = 1e-8):
	''' Find ergodic distribution from a transition matrix

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

def objective_function(rate_guess, market_object):
	
	market_object.set_prices(rate_guess)

	policy = solve_hh(market_object.P, rate_guess, market_object.wage, market_object.tax, market_object.L_tilde, market_object.mu, market_object.gamma, market_object.beta,market_object.delta, market_object.state_grid, market_object.asset_states)

	P = get_transition_matrix(market_object.Q, nb.typed.List(policy), market_object.state_grid)
	P = np.asarray(P) # P.shape = (2000, 2000) = (#states X #states)
	
	#distr = get_distribution(P) # TODO: different guess
	distr_guess = np.full(market_object.state_grid.shape, 1)/len(market_object.state_grid)
	distr = get_distribution_iterate(distr_guess, P, market_object.state_grid)

	K_HH =  distr @ market_object.asset_states[policy]

	loss = (market_object.K - K_HH)**2
	print(market_object.K - K_HH)
	return(loss)


P = np.array([[0.47, 0.53], [0.04, 0.96]])
hh_panel = sim_hh_panel(P, 1000, 2000) # TODO: might not be strictly necessary
	
steady_state = Market(
	hh_panel,
	r = 0.02,
	gamma = 2, # CRRA parameter
	beta = 0.99, # Discount factor
	P = np.array([[0.47, 0.53], [0.04, 0.96]]), # Employment Markov transition matrix
	mu = 0.15, # Unemployment replacement rate
	delta = 0.025, # Capital depreciation rate
	alpha = 0.36, # Capital share
	borrow_constr = 0, # Borrowing constraint
	max_asset = 500,
	T = 1000,
	rho = 0.95,
	sigma = 0.007,
	a_size = 1000) # 1000


sol = minimize_scalar(objective_function, bounds=(0.03, 0.04), method='bounded', args = steady_state)
rate_ss = sol.x

rate_ss = 0.034372948575014266 
steady_state.set_prices(rate_ss)
steady_state.K
policy = solve_hh(steady_state.P, rate_ss, steady_state.wage, steady_state.tax, steady_state.L_tilde, steady_state.mu, steady_state.gamma, steady_state.beta,steady_state.delta, steady_state.state_grid, steady_state.asset_states)
# Found steady state capital: 39.2538 with 1000 grid points
# sol.x = 0.03437
policy_ss = steady_state.asset_states[policy].reshape(2,-1)

objective_function(rate_ss, steady_state)


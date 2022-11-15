import numpy as np
from numba import jit
import quantecon as qe
from quantecon import MarkovChain
import ipdb

class	Market:
	''' Store data and parameterize the incomplete markets economy with aggregate risks
	TODO: tax, 
	'''
	def __init__(self,
		hh_panel,
		r = 0.02,
		w = 1,
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
		self.w = w
		self.tax = 0.1 # TODO calculate this
		self.hh_panel = hh_panel
		self.gamma = gamma
		self.beta = beta
		self.P = P
		self.mu = mu
		self.delta = delta
		self.alpha = alpha
		self.asset_states = np.logspace(start = borrow_constr, stop = np.log10(max_asset+1), num= a_size)-1
		self.sigma = sigma # TODO: what does he mean?
		self.rho = rho
		self.T = T
		self.unemployment_rate = 1 - np.mean(hh_panel) # series of unemployment
		self.L_tilde = 1 / ( 1 - self.unemployment_rate) # How much they work, given that they work. Implies L = 1
		self.L = 1
		self.s_size = 2 * a_size # Number of states (TODO employed unemployed X num assets)
		self.a_size = a_size # number of actions
		self.R = np.empty((self.s_size, self.a_size))
    self.Q = np.zeros((self.s_size, self.a_size, self.s_size))
    self.state_grid = np.asarray(range(2*a_size))
    self.build_R() 
    self.build_Q()


  def set_prices(self, r, w):
      """
      Use this method to reset prices. Calling the method will trigger a
      re-build of R.
      """
      self.r, self.w = r, w
      self.build_R()

  def build_Q(self):
      populate_Q(self.Q, self.s_size, self.a_size, self.state_grid, self.asset_states, self.P)

  def build_R(self):
  	populate_R(
  		self.R, 
  		self.s_size,
  		self.a_size,
  		self.state_grid,
  		self.asset_states,
  		self.r,
  		self.w,
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
  		
  		R[s, a] = (consumption**(1 - gamma) - 1) / (1 - gamma)
  return

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



P = np.array([[0.47, 0.53], [0.04, 0.96]])

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

hh_panel = sim_hh_panel(P, 1000, 2000)
steady_state = Market(
	hh_panel,
	r = 0.02,
	w = 1,
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
	a_size = 1000)



# Use the instance to build a discrete dynamic program
ddp = DiscreteDP(am.R, am.Q, am.β)

# Solve using policy function iteration
results = am_ddp.solve(method='policy_iteration')


def update_beliefs(K, ix):
	K_present = np.log(K[ix])
	K_lead = np.log(K[ix+1])
	slope, intercept, r, p, se = sp.stats.linregress(K_present, y=K_lead)
	coef_new = np.array([intercept, slope])
	return(coef_new, r**2)

def find_eq(belief_guess, model):
	''' iterate on beliefs until fixed point is found
	model is an instance of class Market
	'''
	
	hh_panel = sim_hh_panel(P, 5000, 1000)
	# TODO: need agg shocks as well?
	loss = 100

	while loss > 1e-5:
		hh_policy = find_hh_policy()

		history

		update_beliefs
		
		loss = (belief_guess - beliefs) @ (belief_guess - beliefs)

		belief_guess_new = belief_guess * 0.9 + 0.1 * beliefs

	return(belief_guess_new)




tmp = steady_state.generate_tfp_seq()



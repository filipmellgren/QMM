import quantecon as qe
import numpy as np
from numba import jit, prange

class	Market:
	''' Store data and parameterize the incomplete markets economy with aggregate risks
	'''
	def __init__(self, 
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
		a_size = 1000):
		mc = qe.MarkovChain(P)
		self.r = r
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
		self.unemployment_rate = mc.stationary_distributions[0][0]
		self.L_tilde = 1 / ( 1 - self.unemployment_rate) # How much they work, given that they work. Implies L = 1
		self.L = 1
		self.A = 1 
		self.capital_from_r()
		self.s_size = 2 * a_size # Number of states (TODO employed unemployed X num assets)
		self.a_size = a_size # number of actions
		self.R = np.empty((self.s_size, self.a_size))
		self.Q = np.zeros((self.s_size, self.a_size, self.s_size))
		self.state_grid = np.asarray(range(2*a_size))
		self.get_wage()
		self.get_tax_rate()
		self.build_R() 
		self.build_Q()
	
	def capital_from_r(self):
		''' Capital Demanded, given interest rate.
		'''
		# TODO: Should somethiing else be updated?

		self.K = (self.r/(self.alpha * self.A))**(1/(self.alpha - 1)) * self.L
		

	def r_from_capital(self):
		''' Interest firms can pay, given capital levels
		'''
		self.r = self.alpha * self.A * self.K**(self.alpha - 1) * self.L**(1- self.alpha)

	def set_tfp(self, new_tfp, K):
		''' Shock TFP, guessing new value of K
		Updating K triggers an updating of prices, and therefore also capital
		TODO: should this happen instantaneously though?
		TODO: K is not needed as an argument. It is taken as given when TFP updates and doesn't respond to tfp.
		'''
		self.A = new_tfp
		self.r_from_capital()
		self.get_wage()
		self.build_R()
		# self.set_capital(K)
	
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
	assets = asset_grid[remainder]
	return(assets)

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



def get_dY_t(policy_ss, distr_ss):
	'''
	Y_t in our case is savings at time t
	'''
	P_t = np.empty((T)) # Prediction vector
	for ix in range(T):
		P_t[ix] = np.sum(policy_ss, (distr_ss.T)**(ix))
	
	dD = np.empty((S))


	dY = np.empty((T)) 
	for t in range(T):
		for s in range(S):
			if s > t:
				dY[t] = 0 # TODO: can be made faster
				continue
			dY[t] = P[t-s] * dD[s]

		pass
	pass

def fake_news_matrix(T, S, Y_cal, P_cal, D_cal):
	''' Construct the fake news matrix
	See slide 18 from lecture for definition of caligraphuic letters
	INPUT
		Y_cal : is a numpy array such that dY_t = Y_cal_(s-t), i.e. change in Y at t which is dy_t'D_ss (change in each individual's y holding distribution fixed at ss levels)
	'''
	F = np.empty((T,S))
	for t in range(T):
		for s in range(S):
			if t == 0:
				F[t, s] = Y_cal[s]
				continue
			F[t, s] = P_cal[t-1].T @ D_cal[s]  
	return(F)

def capital_jacobians(fake_news):
	''' Jacobian of the capital function wrt r_s and w_s
	No analytical solution to this one. 
	It is the derivative of household savings wrt r_s and w_s
	'''
	J = np.copy(fake_news.shape)
	T = J.shape[0]
	S = J.shape[1]
	
	for t in range(start = 1, stop = T):
		for s in range(start = 1, stop = S):
			J[t, s] += J[t-1, s-1]	
	return(J)
	







def ir_K():
	''' Change in K wrt z,. the impulse response

	'''
	H_K_inv = np.linalg.inv(H_K)
	dK = - H_K_inv @ H_Z @ dZ # Impulse response
	pass






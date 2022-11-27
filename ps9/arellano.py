# Basic Arellano (2008) model
import matplotlib.pyplot as plt
import numpy as np
import quantecon as qe
import random
import ipdb
from numba import njit, int64, float64, prange
from numba.experimental import jitclass

class Arellano_Economy:
  ''' Stores data and creates primitives for the Arellano economy.
  '''
  def __init__(self, 
	  B_grid_size= 150,   # Grid size for bonds
	  B_grid_min=-2,   # Smallest B value (negative for borrowing)
	  B_grid_max=0,    # Largest B value
	  y_grid_size=2,     # Grid size for income 
	  β=0.8,            # Time discount parameter
	  γ=2,              # Utility parameter
	  r=0.1,            # Lending rate 
	  rho=0.0,            # Persistence in the income process
	  η=0.025,            # Standard deviation of the income process
	  θ=0.0,       # Prob of re-entering financial markets 
	  def_y_param=1.0,# Parameter governing income in default
	  EV_shock = False,
	  EV_prob = 0):  
	  # Save parameters
	  self.β, self.γ, self.r, self.EV_shock, self.EV_prob = β, γ, r, EV_shock, EV_prob
	  self.rho, self.η, self.θ = rho, η, θ

	  self.y_grid_size = y_grid_size
	  self.B_grid_size = B_grid_size
	  self.B_grid = np.linspace(B_grid_min, B_grid_max, B_grid_size)
	  # mc = qe.markov.tauchen(rho, η, 0, 3, y_grid_size)
	  #self.y_grid, self.P = np.exp(mc.state_values), mc.P

	  self.y_grid = np.array([0.2, 1.2])
	  self.P = np.array([[0.2, 0.8], [0.2, 0.8]])
	  # The index at which B_grid is (close to) zero
    self.B0_idx = np.minimum(np.searchsorted(self.B_grid, 1e-10), B_grid_size-1)
    # Output recieved while in default, with same shape as y_grid
    self.def_y = self.y_grid #np.minimum(def_y_param * np.mean(self.y_grid), self.y_grid)
	  
	def params(self):
		return self.β, self.γ, self.r, self.rho, self.η, self.θ, self.EV_shock, self.EV_prob

	def arrays(self):
		return self.P, self.y_grid, self.B_grid, self.def_y, self.B0_idx

@njit
def u(c, γ):
	return c**(1-γ)/(1-γ)

@njit
def compute_q(v_c, v_d, q, params, arrays):
	'''
	Compute the bond price function q(b, y) at each (b, y) pair.

	This function writes to the array q that is passed in as an argument.
	'''
	# Unpack 
	β, _, r, _, _, _,_,_ = params
	P, y_grid, B_grid, _, _ = arrays
	for B_idx in range(len(B_grid)):
		for y_idx in range(len(y_grid)):
			# Compute default probability and corresponding bond price
			default_states = 1.0 * (v_c[B_idx, :] < v_d)
   		delta = np.dot(default_states, P[y_idx, :]) 
    	q[B_idx, y_idx] = (1 - delta ) / (1 + r)

  return(q)


@njit
def T_d(y_idx, v_c, v_d, params, arrays):
  """
  The RHS of the Bellman equation when income is at index y_idx and
  the country has chosen to default.  Returns an update of v_d.
  """
  # Unpack 
  β, γ, _, _, _, θ,_,_ = params 
  P, _, _, def_y, B0_idx = arrays

  current_utility = u(def_y[y_idx], γ)
  
  #v = np.maximum(v_c[B0_idx, :], v_d)
  v = value_beginning(v_c, v_d, B0_idx, False, 0)
  
  cont_value = np.sum((θ * v + (1 - θ) * v_d) * P[y_idx, :])

  return current_utility + β * cont_value

@njit
def value_beginning(v_c, v_d, B_idx, EV_shock, prob):
	if EV_shock:
		v = prob * v_c[B_idx, :] + (1 - prob) * v_d
		return(v)
		
	return(np.maximum(v_c[B_idx, :], v_d))

@njit
def T_c(B_idx, y_idx, v_c, v_d, q, params, arrays):
  '''
  The RHS of the Bellman equation when the country is not in a
  defaulted state on their debt.  Returns a value that corresponds to
  v_c[B_idx, y_idx], as well as the optimal level of bond sales B'.
  '''
  # Unpack 
  β, γ, _, _, _, _, _,_ = params 
  P, y_grid, B_grid, _, _ = arrays
  B = B_grid[B_idx]
  y = y_grid[y_idx]
  # Compute the RHS of Bellman equation
  current_max = -1e10
  Bp_star_idx = 0 # In case all consumption is always negative for some value, choose lowest savings. 
  # Step through choices of next period B'
  for Bp_idx, Bp in enumerate(B_grid):
  	c = y + B - q[Bp_idx, y_idx] * Bp
  	if c > 0:
  		#v = np.maximum(v_c[Bp_idx, :], v_d)
  		v = value_beginning(v_c, v_d, Bp_idx, False, 0)
  		val = u(c, γ) + β * np.sum(v * P[y_idx, :])
  		if val > current_max:
  			current_max = val
  			Bp_star_idx = Bp_idx
  
  return(current_max, Bp_star_idx)

@njit(parallel=True)
def update_values_and_prices(v_c, v_d, B_star, q, params, arrays):
	# Unpack 
	_, _, _, _, _, _,_,_ = params 
	_, y_grid, B_grid, _, _ = arrays
	y_grid_size = len(y_grid)
	B_grid_size = len(B_grid)

	# Compute bond prices and write them to q
	q = compute_q(v_c, v_d, q, params, arrays)

	# Allocate memory
	new_v_c = np.empty_like(v_c)
	new_v_d = np.empty_like(v_d)

	# Calculate and return new guesses for v_c and v_d
	for y_idx in prange(y_grid_size):
		new_v_d[y_idx] = T_d(y_idx, v_c, v_d, params, arrays)
		for B_idx in range(B_grid_size):
	  	new_v_c[B_idx, y_idx], Bp_idx = T_c(B_idx, y_idx, v_c, v_d, q, params, arrays)
	  	B_star[B_idx, y_idx] = Bp_idx

	return new_v_c, new_v_d


def solve(model, tol=1e-8, max_iter=10_000):
  '''
  Given an instance of Arellano_Economy, this function computes the optimal
  policy and value functions.
  '''
  # Unpack
  params = model.params()
  arrays = model.arrays()
  y_grid_size, B_grid_size = model.y_grid_size, model.B_grid_size 

  # Initial conditions for v_c and v_d
  v_c = np.zeros((B_grid_size, y_grid_size))
  v_d = np.zeros(y_grid_size)

  # Allocate memory
  q = np.empty_like(v_c)
  B_star = np.empty_like(v_c, dtype=int)

  current_iter = 0
  dist = np.inf
  while (current_iter < max_iter) and (dist > tol):
  	if current_iter % 100 == 0:
  		print(f"Entering iteration {current_iter}.")
  	new_v_c, new_v_d = update_values_and_prices(v_c, v_d, B_star, q, params, arrays)
  	# Check tolerance and update
  	dist = np.max(np.abs(new_v_c - v_c)) + np.max(np.abs(new_v_d - v_d))
  	v_c = new_v_c
  	v_d = new_v_d
  	current_iter += 1
  print(f"Terminating at iteration {current_iter}.")
  return(v_c, v_d, q, B_star)

Basic_Arellano = Arellano_Economy()

v_c, v_d, q, B_star = solve(Basic_Arellano)

# Create upper envelope
Vf = np.maximum(v_c, v_d)

import pandas as pd
import plotly.express as px

df = pd.DataFrame(Vf, columns = ["low", "high"])
df["B"] = Basic_Arellano.B_grid
df = pd.melt(df, id_vars = "B", var_name = "income")

# Plot the 4 value functions

fig = px.line(df, x = "B", y = "value", color = "income")
fig.show()

q

# q is the discounted expected value of a promise to pay B next period (notation error?)
df = pd.DataFrame(Basic_Arellano.B_grid[B_star]*q, columns = ["low", "high"])
df = pd.DataFrame(q, columns = ["low", "high"])
df["B"] = Basic_Arellano.B_grid 
df = pd.melt(df, id_vars = "B", var_name = "income")
fig = px.line(df, x = "B", y = "value", color = "income")
fig.show()

np.searchsorted(B_grid, 1e-10)


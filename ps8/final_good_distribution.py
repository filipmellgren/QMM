from inv_seq import find_inventory_seq
import numpy as np
from scipy.interpolate import griddata
import scipy as sp
import ipdb

def find_ergodic_distribution(price_guess, params):
	''' Find ergodic distribution using eigenvector approach

	'''
	s_vec, m_vec, adj_vec = find_inventory_seq(price_guess, params)	
	s_grid = griddata(params["inventory_grid"], params["inventory_grid"], xi = s_vec, method = "nearest")
	s_grid = -np.sort(-(np.unique(s_grid)))
	s_grid = np.append(s_grid, s_vec[-1])
	adj_p = adj_vec[0:s_grid.shape[0]-1]
	sfrom, sto = np.meshgrid(s_grid, s_grid[1:], sparse = True, indexing = "ij")
	trans_mat_indic = (sto == sfrom)
	trans_mat_indic = trans_mat_indic.T
	trans_mat_indic = trans_mat_indic[:,0:-1]

	trans_p = (1 - np.expand_dims(np.asarray(adj_p), axis = 1))

	P = trans_mat_indic * trans_p
	P[:,0] = adj_p
	assert np.all(np.isclose(np.sum(P, axis = 1), 1))

	eigen_val, ergodic_distr = sp.sparse.linalg.eigs(P.T, k = 1, sigma = 1)
	ergodic_distr = np.abs(ergodic_distr / np.sum(ergodic_distr))
	ergodic_distr = np.squeeze(ergodic_distr)

	m_vec = np.array(m_vec[0:ergodic_distr.shape[0]])
	s_vec = np.array(s_vec[0:ergodic_distr.shape[0]])
	np.savetxt(f"figures/ergodic_distr_{str(price_guess)[-2:]}.csv", ergodic_distr)
	
	return(ergodic_distr, m_vec, s_grid, s_vec)



def find_ergodic_distribution2(price_guess, params):
	''' Find ergodic distribution using probability a firm has not adjusted

	'''
	s_vec, m_vec, adj_vec = find_inventory_seq(price_guess, params)	
	s_vec, ix = np.unique(s_vec, return_index=True)
	s_vec = s_vec[ix]
	adj_vec = adj_vec[0:s_vec.shape[0]]
	m_vec = m_vec[0:s_vec.shape[0]]
	#adj_p = adj_vec[0:len(s_vec)]
	distr = []
	J_max = len(s_vec)
	for j_ix in range(1,J_max): # start after s_star
		# Solve forward to find mass of firms that haven't adjusted
		not_adjusted = 1 - np.sum(adj_vec[0:j_ix+1]) 
		distr.append(not_adjusted)
	
	
	distr[-1] = distr[-1]/adj_vec[-1]
	ergodic_distr = distr/np.sum(distr)
	
	return(ergodic_distr, np.asarray(m_vec), np.asarray(s_vec), np.asarray(s_vec))


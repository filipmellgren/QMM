# Translated to Python by chatgpt
import numpy as np
import ipdb
from scipy import optimize
import os
# Transition after unexpected decrease in aggregate productivity ("MIT shock")

ga = 2
rho = 0.05
rho = 0.005
d = 0.05
d = 0.025
al = 1/3
Aprod = 0.1
z1 = 1
z2 = 2*z1
z = np.array([z1,z2])
la1 = 1/3
la2 = 1/3
la = np.array([la1,la2])
z_ave = (z1*la2 + z2*la1)/(la1 + la2)

T = 200
# N = 400
N = 800
dt = T/N
time = np.arange(0, N)*dt
max_price_it = 300
convergence_criterion = 10^(-5)

# Asset grid
I= 1000
amin = -0.8
# amin = 0.0
amax = 20
a = np.linspace(amin, amax, I)
da = (amax-amin)/(I-1)

aa = np.array([a, a]).T
zz = np.ones((I,1))*z

# Some convergence parameters
maxit= 100
crit = 10^(-6)
Delta = 1000
Ir = 40
crit_S = 10^(-5)

# Declarations
dVf = np.zeros((I,2))
dVb = np.zeros((I,2))
c = np.zeros((I,2))

# Create the Aswitch matrix
# TODO: is this correct?
Aswitch = np.concatenate([-np.eye(I) * la[0], np.eye(I) * la[0]], axis=1)
Aswitch = np.concatenate([Aswitch, np.concatenate([np.eye(I) * la[1], -np.eye(I) * la[1]], axis=1)])

# Generate initial values
r = 0.04
w = 0.05
v0 = np.column_stack([(w * z[0] + r * a)**(1 - ga) / (1 - ga) / rho,
                      (w * z[1] + r * a)**(1 - ga) / (1 - ga) / rho])

# Starting point, upper and lower bound for interest rate bisection
r0 = 0.03
rmin = 0.0001
rmax = 0.99 * rho

## Steady State
def solve_firm(r, al, Aprod, d, z_ave):
    KD = (al * Aprod / (r + d)) ** (1 / (1 - al)) * z_ave
    w = (1 - al) * Aprod * KD ** al * z_ave ** (-al)
    return(KD, w)

def solve_hh(r, v, dVb, dVf, I, w, z, amin, amax, ga, da, zz, aa, Aswitch, Delta, rho):
    maxit = 100
    for n in range(maxit): 
        # Solve HH problem
        V = np.copy(v)
        
        # forward difference
        dVf[:I - 1, :] = (V[1:I, :] - V[:I - 1, :]) / da
        
        dVf[I-1, :] = (w * z + r * amax) ** (-ga)  # will never be used, but impose state constraint a <= amax just in case

        # backward difference
        # TODO: this is the same as the forward difference in the matlab code. Correct?
        dVb[1:I, :] = (V[1:I, :] - V[:I - 1, :]) / da
        dVb[0, :] = (w * z + r * amin) ** (-ga)  # state constraint boundary condition

        # consumption and savings with forward difference
        cf = dVf ** (-1 / ga)
        ssf = w * zz + r * aa - cf # TODO: not sure if elementwise or matrix product (aa was transposed by me)

        # consumption and savings with backward difference
        cb = dVb ** (-1 / ga)
        ssb = w * zz + r * aa - cb

        # when backward and forward difference is not applicable, we use steady state
        # below is consumption and derivative of value function at steady state
        c0 = w * zz + r * aa
        dV0 = c0 ** (-ga)

        # `dV_Upwind` makes a choice of forward or backward differences based on
        # the sign of the drift
        # indicators
        If = ssf > 0  # positive drift --> forward difference
        Ib = ssb < 0  # negative drift --> backward difference
        I0 = (1 - If - Ib)  # at steady state

        # upwind and consumption
        dV_Upwind = dVf * If + dVb * Ib + dV0 * I0  # important to include third term
        c = dV_Upwind ** (-1 / ga)
        u = c ** (1 - ga) / (1 - ga)

        # construct a matrix
        X = -np.minimum(ssb, 0) / da
        Y = -np.maximum(ssf, 0) / da + np.minimum(ssb, 0) / da
        Z = np.maximum(ssf, 0) / da

        # construct matrix A
        # TODO: Looks like these need extra work.
        # S = spdiags(Bin,d,m,n) creates an m-by-n sparse matrix S by taking the columns of Bin and placing them along the diagonals specified by d. 
        # TODO: so ensure A1 is I by I
        # Should probably use np.diagflat(Y[:,0], 0)
        # https://numpy.org/doc/stable/reference/generated/numpy.diagflat.html#numpy.diagflat
        A1 = (
            np.diag(Y[:, 0])
            + np.diag(X[1:I, 0], -1)
            + np.diag(Z[:I - 1, 0], 1)
        )
        A2 = (
            np.diag(Y[:, 1])
            + np.diag(X[1:I, 1], -1)
            + np.diag(Z[:I - 1, 1], 1)
        )
        # First zero matrices in NE, SW. Then add Aswitch (2000 by 2000)
        A = np.block([[A1, np.zeros((I, I))], [np.zeros((I, I)), A2]]) + Aswitch
        
        
        # construct matrix B
        B = (1 / Delta + rho) * np.eye(2 * I) - A

        # stack matrices and solve system of equations
        u_stacked = np.hstack((u[:, 0], u[:, 1]))
        V_stacked = np.hstack((V[:, 0], V[:, 1]))
        b = u_stacked + V_stacked / Delta
        
        V_stacked = np.linalg.solve(B, b)

        # update value
        V = V_stacked.reshape((I, 2), order = "F") 

        # compute critical value
        Vchange = V - v
        v = np.copy(V)

        dist = np.max(np.abs(Vchange))

        # check convergence
        if dist < crit:
            print("Value Function Converged, Iteration = ")
            print(n)
            break
    return(A)

def solve_distr(A, I, da):
    AT = np.copy(A.T)
    b = np.zeros(2*I)
    # need to fix one value, otherwise matrix is singular
    i_fix = 0
    b[i_fix] = 1
    # TODO: not sure what this line did exactly. Deleted and incorporated belwo
    AT[i_fix,:] = 1 # TODO: hacky, was 0, changed to 1 to avoid singulairty problem
    AT[i_fix,0] = 1

    # Solve linear system
    gg = np.linalg.solve(AT, b) # TODO: still singular (rank 1999)
    g_sum = np.dot(gg.T, np.ones((2*I,1))*da)
    gg = gg / g_sum

    g = np.concatenate((gg[:I], gg[I:2*I]))
    return(g)

def check_mc(r, al, Aprod, d, z_ave, v0, dVb, dVf, I, w, z, amin, amax, ga, da, zz, aa, Aswitch, Delta, rho, a):
    
    KD, w = solve_firm(r, al, Aprod, d, z_ave)
    A = solve_hh(r, v0, dVb, dVf, I, w, z, amin, amax, ga, da, zz, aa, Aswitch, Delta, rho)
    g = solve_distr(A, I, da)
    KS = np.dot(g, np.tile(a*da, 2))
    return(KS - KD)

path = 'r_ss.txt'
if not os.path.exists(path):
  r_ss = optimize.bisect(lambda x: check_mc(x, al, Aprod, d, z_ave, v0, dVb, dVf, I, w, z, amin, amax, ga, da, zz, aa, Aswitch, Delta, rho, a), 0.001, 0.01)
  f = open(path, "w")
  f.write(str(r_ss))
  f.close()

f = open(path, "r")
r_ss = f.read()
f.close()
r_ss = float(r_ss)




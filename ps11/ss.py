# STEADY STATE
import ipdb
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
        np.max(A1)
        
        
        # construct matrix B
        B = (1 / Delta + rho) * np.eye(2 * I) - A

        # stack matrices and solve system of equations
        u_stacked = np.hstack((u[:, 0], u[:, 1]))
        V_stacked = np.hstack((V[:, 0], V[:, 1]))
        b = u_stacked + V_stacked / Delta
        ipdb.set_trace()
        V_stacked = np.linalg.solve(B, b)
        
        print(b)

        # update value
        V = V_stacked.reshape((I, 2), order = "F") # TODO: look into this line by chat gpt

        # compute critical value
        Vchange = V - v
        v = V

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
    ipdb.set_trace()
    g = solve_distr(A, I, da)
    KS = np.dot(g, np.tile(a*da, 2))
    return(KS - KD)

check_mc(r, al, Aprod, d, z_ave, v0, dVb, dVf, I, w, z, amin, amax, ga, da, zz, aa, Aswitch, Delta, rho, a)

# loop over `Ir`
ir = 0
for ir in range(Ir):
    # TODO: maybe I can skip this loop and apply bisection directly

    KD, w = solve_firm(r, al, Aprod, d, z_ave)

    # if `ir` is greater than 1, set `v0` to the previous iteration's value
    # TODO: is this step just here to speed up computation? I think so because we then iterate on v until convergence to fixed point. 
    #if ir > 1:
    #   v0 = V_r[:, :, ir - 1]

    v = v0

    # loop over `maxit`
    for n in range(maxit): 
        # Solve HH problem
        V = v
        # forward difference
        dVf[:I - 2, :] = (V[1:I-1, :] - V[:I - 2, :]) / da
        
        dVf[I-1, :] = (w * z + r * amax) ** (-ga)  # will never be used, but impose state constraint a <= amax just in case

        # backward difference
        # TODO: this is the same as the forward difference in the matlab code. Correct?
        dVb[1:I-1, :] = (V[1:I-1, :] - V[:I - 2, :]) / da
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
        V = V_stacked.reshape((I, 2)) # TODO: look into this line by chat gpt

        # compute critical value
        Vchange = V - v
        v = V

        dist = np.max(np.abs(Vchange))

        # check convergence
        if dist < crit:
            print("Value Function Converged, Iteration = ")
            print(n)
            break

    # Distribution
    g = solve_distr(A, I, da)

    # Aggregating assets
    KS = np.dot(g, np.tile(a*da, 2))

    # Market clearing
    KS - KD

    # Update value function
    # TODO: remember that this needs to be passed. Or does it need to be passed?
    v0 = v



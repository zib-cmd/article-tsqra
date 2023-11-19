import numpy as np
import scipy

def generate_indices(xbins, ybins):
    
    s = np.zeros((xbins * ybins, 3))
    k = 0
    for j in range(ybins):
        for i in range(xbins):
            s[k,0] = i
            s[k,1] = j
            s[k,2] = i + ybins * j
            k     = k + 1
    
    return s

def generate_indices_2(xbins, ybins):
    return np.c_[np.array(np.meshgrid(np.arange(ybins), np.arange(xbins))).T.reshape(-1,2), np.arange(xbins * ybins)]

    
def generate_tildeQrows(Q1,Q2,sqrt_pi12,i,j,xbins,ybins):
    """
    tildeQrows = generate_tildeQrows(Q1,Q2,sqrt_pi12,i,j,ybins)
    """
    Nreps = i.shape[0]
    
    # current state
    s  = i + ybins * j
    
    s1_plus  = (i + 1) + ybins * j
    s1_minus = (i - 1) + ybins * j
    s2_plus  =    i    + ybins * ( j + 1 )
    s2_minus =    i    + ybins * ( j - 1 ) 
    
    tildeQrows = np.zeros((Nreps, xbins * ybins))
    
    # fancy indexing
    tildeQrows[np.arange(Nreps),s1_plus]  = Q1[i,i+1] * sqrt_pi12[s1_plus]  / sqrt_pi12[s]
    tildeQrows[np.arange(Nreps),s1_minus] = Q1[i,i-1] * sqrt_pi12[s1_minus] / sqrt_pi12[s]
    tildeQrows[np.arange(Nreps),s2_plus]  = Q2[j,j+1] * sqrt_pi12[s2_plus]  / sqrt_pi12[s]
    tildeQrows[np.arange(Nreps),s2_minus] = Q2[j,j-1] * sqrt_pi12[s2_minus] / sqrt_pi12[s]
    tildeQrows[np.arange(Nreps),s]        = - np.sum(tildeQrows, axis=1)
    
    return tildeQrows
    
def rates_gillespie(Q,n):
    """
    Q : Rate matrix
    n : row to select
    norm_rates, sr, cdf = rates_gillespie(Q,n)
    """
    
    # Select the row
    r          = Q[n,:]
    
    # Take the absolute value
    r          = np.abs(r)
    
    # Calculate the sum
    sr         = np.sum(r, axis=1)
    
    # Normalize
    norm_rates = r / sr[:,None]
       
    # CDF
    cdf  = np.cumsum(norm_rates, axis=1)
        
    return norm_rates, sr, cdf

def perturbed_rates_gillespie(tildeQrows):
    """
    Q : Rate matrix
    norm_rates, sr, cdf = perturbed_rates(tildeQrows)
    """

    # Select the rows
    r          = tildeQrows

    # Take the absolute value
    r          = np.abs(r)

    # Calculate the sum
    sr             = r.sum(axis=1)
    invsr          = scipy.sparse.diags(1/sr.ravel())

    # Normalize
    norm_rates = invsr @ r

    # CDF
    cdf  = norm_rates.copy()

    indptr = cdf.indptr
    data   = cdf.data
    for i in range(cdf.shape[0]):
        st = indptr[i]
        en = indptr[i + 1]
        np.cumsum(data[st:en], out=data[st:en])
        
    return norm_rates, sr, cdf

def trans_probs_time_driven(tildeQrows, s, dt):
    """
    Q : Rate matrix
    t_probs = trans_probs_time_driven(tildeQrows, s, dt)
    """
    Nreps = tildeQrows.shape[0]
    
    # Identiy rows
    I = np.zeros(tildeQrows.shape)
    I[np.arange(Nreps),s] = 1

    # Select the rows
    t_probs = I + dt * tildeQrows
    
    return t_probs

def jump_gillespie(cdf):
    """
    cdf   : cumulative distribution function of rates between 0 and 1
    reaction = jump_gillespie(cdf)
    """
    Nreps = cdf.shape[0]
    
    # Draw a random number from a uniform distribution
    u   = np.random.uniform(0, 1, Nreps)        

    # Select reaction
    reaction = np.sum(u[:,None] > cdf, axis = 1)
     
    return reaction
                
def jump_time_driven(t_probs):
    """
    cdf   : cumulative distribution function of rates between 0 and 1
    reaction = jump_time_driven(t_probs)
    """
    Nreps = t_probs.shape[0]
    
    # Draw a random number from a uniform distribution
    u   = np.random.uniform(0, 1, Nreps)        

    cdf = np.cumsum(t_probs)
    
    # Select reaction
    reaction = np.sum(u[:,None] > cdf, axis = 1)
        
    return reaction    
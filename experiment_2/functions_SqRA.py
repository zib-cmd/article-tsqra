import numpy as np
from scipy import stats
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

def grid1(xmin, xmax, xbins):
    xedges   = np.linspace(xmin, xmax, xbins) 
    dx       = xedges[1] - xedges[0]
    xcenters = xedges + 0.5 * dx
    xcenters = np.delete(xcenters,-1)
    xbins    = len(xcenters)
    
    return xcenters, xedges, xbins, dx

def adjancency_matrix_sparse(bins, flux, periodic=False):
    # bins : array with the number of bins for each dimension
    # flux : array with the flux in each direction
    # For example:
    #
    # 1D
    # bins = np.array([xbins])
    # flux = np.array([D/dx**2])
    #
    # 2D
    # bins = np.array([xbins,   ybins])
    # flux = np.array([D/dx**2, D/dy**2])
        
    # Number of dimensions
    Nd = bins.shape[0]
    
    A = 0
    for i in range(0, Nd):

        v = np.zeros(bins[i])
        v[1] = 1

        if periodic:
            v[-1] = 1
            Ai = scipy.sparse.csc_matrix(scipy.linalg.circulant(v)) #.toarray()
        else:
            Ai = scipy.sparse.csc_matrix(scipy.linalg.toeplitz(v)) #.toarray()

        A = scipy.sparse.kronsum(A, flux[i] * Ai)
        
    return A

def build_Qo(w, A, xbins):
    SQRA  = np.sqrt(w.astype(float))
    Di    = scipy.sparse.spdiags( SQRA,     0, xbins, xbins) #.toarray()
    Dinv  = scipy.sparse.spdiags( 1 / SQRA, 0, xbins, xbins)  

    return Dinv *  A * Di, Dinv, Di

def build_sqra(D, w, dx, bins):
    
    # Total number of bins
    Nbins = np.prod(bins)

    # Flux (array)
    flux = D / dx**2    
    
    # Adjacency matrix
    A  = adjancency_matrix_sparse(bins, flux, periodic=False)
    
    # Diagonalization
    Qo = build_Qo(w, A, Nbins)[0]

    E = scipy.sparse.spdiags( Qo.sum(axis=1).T, 0, Nbins, Nbins )
    Q     = Qo - E 

    # Dense matrix
    Qd = Q.toarray()

    #chi, e, S, schurevecs = pcca(Qd, 2, massmatrix=None)
    #Qc = np.linalg.pinv(chi).dot(Qd.dot(chi))
    
    return Qd, Qo #Qc, chi, Qo
    
def build_sqra_i(D, W, DX, bins):
       
    # Number of dimensions
    Nd    = W.shape[0]
    
    # Total number of bins
    Nbins = np.prod(bins)
    
    # Adjacency matrix
    Ai  = np.array([adjancency_matrix_sparse(np.array([bins[i]]), flux = np.array([D / DX[i] ** 2]), periodic=False) for i in range(Nd)])
    
    Qi = np.empty(Nd, dtype=object)
    Ai  = np.empty(Nd, dtype=object)
    Ii  = np.empty(Nd, dtype=object)
    Qoi = np.empty(Nd, dtype=object)
    Ei  = np.empty(Nd, dtype=object)
    
    A   = 0
    Qo  = 0
    E   = 0
    
    for i in range(Nd):
        
        Ai[i]  = adjancency_matrix_sparse(np.array([bins[i]]), flux = np.array([D / DX[i] ** 2]), periodic=False)
        Ii[i]  = scipy.sparse.eye(*Ai[i].shape)
        Qoi[i] = build_Qo( W[i], Ai[i], bins[i] )[0]
        Ei[i]  = scipy.sparse.spdiags( Qoi[i].sum(axis=1).T, 0, bins[i], bins[i] )
        Qi[i]  = Qoi[i] - Ei[i]

    
    """    
        A      = scipy.sparse.kronsum( A,  Ai[i]  )
        Qo     = scipy.sparse.kronsum( Qo, Qoi[i] ) 
        E      = scipy.sparse.kronsum( E,  Ei[i])
    
    #E = scipy.sparse.spdiags( Qo.sum(axis=1).T, 0, Nbins, Nbins )
    Q     = Qo - E 
    
    # Dense matrix
    Qd = Q.toarray()

    chi, e, S, schurevecs = pcca(Qd, 2, massmatrix=None)
    Qc = np.linalg.pinv(chi).dot(Qd.dot(chi))
    """
    return Qi, Qoi

    
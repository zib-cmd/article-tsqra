import numpy as np
from scipy.linalg import schur, ordqz
from optimization import inner_simplex_algorithm, optimize

def cluster_by_isa(Evs, NoOfClus = 2):
    
    if NoOfClus < 2:
        Chi      =  np.ones(Evs.shape[0])
        indic    = 0
        RotMat   = 1 / Evs[0,0]
        cF       = np.ones(Evs.shape[0])
    elif NoOfClus >= Evs.shape[0] and Evs.shape[1] == Evs.shape[0]:
        Chi      = np.eye(Evs.shape[0])
        indic    = 0
        RotMat   = npp.linalg.inv(Evs)
        cF       = np.arange(Evs.shape[0])
    else:
        NoOfClus = np.round(NoOfClus)
        
        if NoOfClus > Evs.shape[1]:
            NoOfClus = Evs.shape[1]
        
        C        = Evs[:, 0:NoOfClus]
        
        OrthoSys = np.copy(C)
        maxdist  = 0.0
    
        ind = np.zeros(NoOfClus, dtype=int)
        for i in range(Evs.shape[0]): 
            if np.linalg.norm(C[i,:]) > maxdist:
                maxdist = np.linalg.norm(C[i,:])
                ind[0]  = i
        
        for i in range(Evs.shape[0]): 
            OrthoSys[i,:] = OrthoSys[i,:] - C[ind[0],:]
        
        for k in range(1, NoOfClus):
            maxdist = 0.0
            temp    = np.copy(OrthoSys[ind[k-1],:])

            for i in range(Evs.shape[0]): 
                
                OrthoSys[i,:] = OrthoSys[i,:] - np.matmul(temp , OrthoSys[i,:].T) * temp 
                distt         = np.linalg.norm(OrthoSys[i,:])
                #print(distt)
                if distt > maxdist:
                    maxdist = np.copy(distt)
                    ind[k]  = i
                    
            OrthoSys = OrthoSys / np.linalg.norm(OrthoSys[ind[k],:])
        
        
        RotMat = np.linalg.inv(C[ind,:])
        Chi    = np.matmul( C, RotMat )
        indic  = np.min(Chi)
        #[minVal cF] = np.max(transpose(Chi))
    
        
    return Chi, RotMat

def pcca(T, n, massmatrix=None):
    X, e = schurvects(T, n, massmatrix)
    A = inner_simplex_algorithm(X)
    if n > 2:
        A = optimize(X, A)
    chi = X.dot(A)
    return chi, e, A, X


def schurvects(T, n, massmatrix=None):
    e = np.sort(np.linalg.eigvals(T))

    v_in  = np.real(e[-n])
    v_out = np.real(e[-(n + 1)])

    # do not seperate conjugate eigenvalues
    assert not np.isclose(v_in, v_out), \
        "Cannot seperate conjugate eigenvalues, choose another n"

    # determine the eigenvalue gap
    cutoff = (v_in + v_out) / 2

    # schur decomposition
    if massmatrix is None:
        _, X, _ = schur(T, sort=lambda x: np.real(x) > cutoff)
    else:
        _, _, _, _, _, X = \
            ordqz(T, massmatrix, sort=lambda a, b: np.real(a / b) > cutoff)

    X = X[:, 0:n]  # use only first n vectors

    # move constant vector to the front, make it 1
    X /= np.linalg.norm(X, axis=0)
    i = np.argmax(np.abs(np.sum(X, axis=0)))
    X[:, i] = X[:, 0]
    X[:, 0] = 1
    # return selected Schurvecs and sorted values
    return X, e
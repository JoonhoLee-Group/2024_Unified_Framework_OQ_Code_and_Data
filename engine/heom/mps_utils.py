import numpy as np

def mapint(i, N):
    if i < 0:
        i += N
    if i < 0 or  i >= N:
        raise IndexError("The index (%d) is out of range." % i)
    return i

def determine_truncation(S, tol = None, nbond = None, chimin = None, is_ortho = True):
    nsvd = S.shape[0]

    #we will only allow for truncation if we are currently at the orthogonality centre
    if is_ortho:
        if not nbond == None:
            if nsvd > nbond:
                nsvd = nbond

        if not tol == None:
            #print(np.amax(S), tol)
            #print(S/np.amax(S))
            ntrunc = np.argmax(S < tol*np.amax(S))
            if(ntrunc == 0):
                ntrunc = nsvd
            if nsvd > ntrunc:
                nsvd = ntrunc

        if not chimin == None:
            if nsvd < chimin:
                nsvd = chimin

    return nsvd


def two_site_matrix_to_tensor(M, d1, d2):
    return np.transpose(M.reshape((d1, d2, d1, d2)), [0, 2, 1, 3])

#functions used for constructing two site object
def two_site_mpo(M, d1, d2, order = 'mpo', tol=None):
    Mmt = two_site_matrix_to_tensor(M, d1, d2)
    Mm = Mmt.reshape((d1*d1, d2*d2))

    Q, S, Vh = np.linalg.svd(Mm, full_matrices=False, compute_uv=True)
    nsvd = determine_truncation(S, tol=tol)

    Q = Q[:, :nsvd]
    S = np.diag(S[:nsvd])
    Vh = Vh[:nsvd, :]
    R = Q @ S
    if(order == 'mpo'):
        Rt = R.reshape((1, d1, d1, nsvd))
        Vht = Vh.reshape((nsvd, d2, d2, 1))
    else:
        Rt = np.transpose(R.reshape((1, d1, d1, nsvd)), [0, 1, 3, 2])
        Vht = Vh.reshape((nsvd, d2, 1, d2))
    return Rt, Vht


def permute_nsite_dims(M, ind, ds):
    indms = [ [x, i] for i,x  in enumerate(ind)]
    #for i, m in enumerate(mi):
    #    indms.append([m, i])
    
    
    #sort the list based on the values of m
    indms = sorted(indms, key = lambda x : x[0])
    pind = [ims[0] for ims in indms]

    perms = [ims[1] for ims in indms]

    pdms = [ds[x] for x in perms]
    
    for ims in indms:
        perms.append(ims[1]+len(ds))
    
    Mtens = np.transpose(M.reshape((*ds, *ds)), axes=perms)
    return Mtens, pind, pdms


def nsite_tensor_to_mpo(M, d):
    #permute the matrix into a tensor with indices the same as needed for an MPO
    xs = [val for pair in zip(range(len(d)), range(len(d), 2*len(d))) for val in pair]
    #now we need to permute this into the mpo ordering
    return np.transpose(M, axes=xs)


def nsite_mpo(M, d, order = 'mpo', tol=None):
    Mt = nsite_tensor_to_mpo(M, d)

    mpo = []

    d1 = 1
    d2 = int(np.prod(d))**2

    nsvdp = 1

    #now we need to iterate over each of the indices of Mt and 
    for i in range(len(d)-1):
        d1 = nsvdp * (d[i]**2)
        d2 = d2 // (d[i]**2)
        Q, S, Vh = np.linalg.svd(Mt.reshape((d1, d2)), full_matrices=False, compute_uv=True)
        nsvd = determine_truncation(S, tol=tol)

        Q = Q[:, :nsvd]
        S = np.diag(S[:nsvd])
        Vh = Vh[:nsvd, :]
        Mt = S @ Vh

        d1 = nsvd
        if order == 'mpo':
            mpo.append(Q.reshape((nsvdp, d[i], d[i], nsvd)))
        else:
            mpo.append(np.transpose(Q.reshape((nsvdp, d[i], d[i], nsvd)), [0, 1, 3, 2]))

        if(i + 2 == len(d)):
            if(order == 'mpo'):
                mpo.append(Mt.reshape(nsvd, d[-1], d[-1], 1))
            else:
                mpo.append(Mt.reshape(nsvd, d[-1], 1, d[-1]))
        nsvdp=nsvd
    return mpo


def identity_pad_mpo(nsvd, d, order = 'mpo'):
    if(order == 'mpo'):
        return np.transpose(np.identity((nsvd*d)).reshape(nsvd, d, nsvd, d), [0, 1, 3, 2])
    else:
        return np.identity((nsvd*d)).reshape(nsvd, d, nsvd, d)


def check_compatible(A, B, Atype, Btype):
    if not (isinstance(A, Atype) and isinstance(B, Btype)):
        raise ValueError("Unable to contract two non-mps object.")
    if(not A.nsites() == B.nsites()):
        raise ValueError("Unable to contract mps object with different number of sites.")

    if(A.local_dimension(slice(0, len(A))) != B.local_dimension(slice(0, len(A)))):
        raise ValueError("Unable to contract mps objects with different local hilbert space dimensions.")


#functions used to shift the orthogonality centre of an mps or mpo
def update_mps_ortho_centre(il, ir, oc, dir):
    if dir == 'right':
        if oc == il:
            return ir
        else:
            return oc   
    elif dir == 'left':
        if oc == ir:
            return il
        return oc
    return oc


def local_canonical_form(Mi, Mj, dir, il, oc, tol = None, nbond = None, chimin = None):
    #if dir == 'right' we need to reshape and svd the left tensor
    if dir == 'right':
        dims = Mi.shape
        A = Mi.reshape((dims[0]*dims[1], dims[2]))
    
        Q = None; R = None
        if tol == None and nbond == None:
            Q, R = np.linalg.qr(A, mode='reduced')
    
        else:
            Q, S, Vh = np.linalg.svd(A, full_matrices=False, compute_uv=True)
    
            nsvd = determine_truncation(S, tol = tol, nbond = nbond, chimin=chimin, is_ortho = (il == oc))

    
            Q = Q[:, :nsvd]
            S = np.diag(S[:nsvd])
            Vh = Vh[:nsvd, :]
            R = S @ Vh
    
        return Q.reshape((dims[0], dims[1], R.shape[0])), R, Mj
    
    #otherwise we reshape and svd the right tensor
    elif dir == 'left':
        dims = Mj.shape
        B = Mj.reshape((dims[0], dims[1]*dims[2]))
    
        Vh = None; R = None
        if tol == None and nbond == None:
            U, L = np.linalg.qr(B.T, mode='reduced')
            Vh = U.T
            R = L.T
    
        else:
            Q, S, Vh = np.linalg.svd(B, full_matrices=False, compute_uv=True)
    
            nsvd = determine_truncation(S, tol = tol, nbond = nbond, chimin=chimin, is_ortho = (il + 1 == oc))
    
            Q = Q[:, :nsvd]
            S = np.diag(S[:nsvd])
            Vh = Vh[:nsvd, :]
            R = Q @ S
    
        return Mi, R, Vh.reshape( (R.shape[1], dims[1], dims[2]))
    
    else:
        ValueError("Invalid dir argument")

def shift_mps_bond(Mi, Mj, dir, il, oc, tol = None, nbond = None, chimin = None):
    X, R, Y = local_canonical_form(Mi, Mj, dir, il, oc, tol=tol, nbond=nbond, chimin=chimin)
    if dir == 'right':
        return X, np.tensordot(R, Y, axes=([1], [0]))
    #otherwise we reshape and svd the right tensor
    elif dir == 'left':
        return np.tensordot(X, R, axes=([2], [0])), Y
    else:
        ValueError("Invalid dir argument")

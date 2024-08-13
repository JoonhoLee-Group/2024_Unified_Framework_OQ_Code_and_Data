import numpy as n
import copy
from ..mps.utils import *
from scipy.linalg import svd, qr, rq
import time

#a basic shared memory mps implementation.  This contains a number 
class adt:
    def __init__(self, rho0):
        self._tensors = []
        self._D2 = rho0.shape[0]
        self._tensors.append(rho0.reshape((1, self._D2, 1)))
        self._orth_centre = 0
        self._is_valid = True
        self._is_conjugated = False

    #access tensor by index
    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[ii] for ii in range(*i.indices(len(self)))]
        elif isinstance(i, int):    
            if i < 0:
                i += len(self)
            if i < 0 or  i >= len(self):
                raise IndexError("The index (%d) is out of range." % i)
            if self._is_conjugated:
                return np.conj(self._tensors[i])
            else:
                return self._tensors[i]
        else:
            raise TypeError("Invalid argument type")
            return None

    def __getslice__(self, i):
        return self.__getitem__(i)

    #set tensor by index
    def __setitem__(self, i, v):
        #first we check that v is a valid type
        if not isinstance(v, np.ndarray):
            raise TypeError("Invalid type for setting item.")
        if not v.ndim == 3:
            raise TypeError("Invalid type for setting item.")

        if isinstance(i, slice):
            for ii in range(*i.indices(len(self))):
                self._tensors[ii] = v
        elif isinstance(i, int):    
            if i < 0:
                i += len(self)
            if i < 0 or  i >= len(self):
                raise IndexError("The index (%d) is out of range." % i)
            self._tensors[i] = v
        else:
            raise TypeError("Invalid argument type")

        self._orth_centre = None
        self._is_valid = None
        self.is_valid()

    def __setslice__(self, i, v):
        self.__setitem__(i, v)

    def is_ortho(self):
        return not self._orth_centre == None

    def __len__(self):
        return len(self._tensors)

    def nsites(self):
        return len(self)

    def nbonds(self):
        return len(self)-1

    def conj(self):
        ret._tensors = self._tensors
        ret._orth_centre = self._orth_centre
        ret._is_valid = self._is_valid
        ret._is_conjugated = not self._is_conjugated
        return ret

    def isconjugated(self):
        return self._is_conjugated

    def shape(self, i):
        if isinstance(i, slice):
            return [self.shape(ii) for ii in range(*i.indices(len(self)))]
        elif isinstance(i, int):    
            if i < 0:
                i += len(self)
            if i < 0 or  i >= len(self):
                raise IndexError("The index (%d) is out of range." % i)
            return self._tensors[i].shape
        else:
            raise TypeError("Invalid argument type")

    def local_dimension(self, i):
        if isinstance(i, slice):
            return [self.local_dimension(ii) for ii in range(*i.indices(len(self)))]
        elif isinstance(i, int):    
            if i < 0:
                i += len(self)
            if i < 0 or  i >= len(self):
                raise IndexError("The index (%d) is out of range." % i)
            return self._tensors[i].shape[1]
        else:
            raise TypeError("Invalid argument type")

    def bond_dimension(self, i):
        if isinstance(i, slice):
            return [self.bond_dimension(ii) for ii in range(*i.indices(len(self)))]
        elif isinstance(i, int):    
            i = mapint(i, self.nbonds())
            return self._tensors[i].shape[2]
        else:
            raise TypeError("Invalid argument type")

    def maximum_bond_dimension(self):
        bd = 0
        for i in range(self.nbonds()):
            if self._tensors[i].shape[2] > bd:
                bd = self._tensors[i].shape[2]
        return bd

    def expand_bond(self, i, chi):
        if isinstance(i, int):    
            i = mapint(i, self.nbonds())
            if not self.shape(i)[2] == self.shape(i+1)[0]:
                raise RuntimeError("Cannot expand bond, the specified bond is invalid.")
            if(chi > self.shape(i)[2]):
                npad = chi - self.shape(i)[2]
                self._tensors[i] = np.pad(self._tensors[i], ((0,0), (0, 0), (0,npad) ), 'constant', constant_values=(0))
                self._tensors[i+1] = np.pad(self._tensors[i+1], ((0,npad), (0, 0), (0,0) ), 'constant', constant_values=(0))
        else:
            raise TypeError("Invalid argument type")

    #function for ensuring that the mps is currently valid
    def is_valid(self):
        if not self._is_valid == None:
            return self._is_valid
        else:
            iv = True
            for i in range(len(self._tensors)):
                if not self.shape(i-1)[2] == self.shape(i)[0]:
                    iv = False
            self._is_valid = iv
            return self._is_valid

    def orthogonalise(self):
        for i in reversed(range(self.nbonds())):
            self.svd_bond(i, dir='left')
        self._orth_centre = 0

    def shift_orthogonality(self, i, tol = None, nbond = None):
        i = mapint(i, self.nsites())
        if self._orth_centre == None:
            self.orthogonalise()

        if i < self._orth_centre:
            for bi in reversed(range(i, self._orth_centre)):
                self.shift_left(tol = tol, nbond = nbond)

        elif i > self._orth_centre:
            for bi in range(self._orth_centre, i):
                self.shift_right(tol = tol, nbond = nbond)

        if(i != self._orth_centre):
            raise RuntimError("the orthogonality centre has not been shifted to the correct position.")


    def normalise(self):
        norm = None
        if(self._orth_centre != None):
            oc = self._orth_centre
            norm = np.dot(np.conj(self._tensors[oc].flatten()), self._tensors[oc].flatten())
            self._tensors[oc] /= np.sqrt(norm)
        else:
            norm = contract(self, self)
            self._tensors[0] /= np.sqrt(norm)
        return norm

    def shift_left(self, tol = None, nbond = None):
        if not self.is_valid():
            raise RuntimeError("The object does not represent a valid MPS.  Unable to perform transformation operations to the MPS.")
        if self._orth_centre == None:
            raise RuntimeError("The object does not have an orthogonality centre to shift")
        if self._orth_centre == 0:
            raise RuntimeError("Orthogonality Centre cannot be shifted left")
        self.shift_bond(self._orth_centre - 1, dir='left', tol = tol, nbond = nbond)

    def shift_right(self, tol = None, nbond = None):
        if not self.is_valid():
            raise RuntimeError("The object does not represent a valid MPS.  Unable to perform transformation operations to the MPS.")
        if self._orth_centre == None:
            raise RuntimeError("The object does not have an orthogonality centre to shift")
        if self._orth_centre == self.nbonds():
            raise RuntimeError("Orthogonality Centre cannot be shifted left")
        self.shift_bond(self._orth_centre , dir='right', tol = tol, nbond = nbond)


    def shift(self, dir, tol = None, nbond = None):
        if dir == 'left':
            self.shift_left(tol=tol, nbond=nbond)
        elif dir == 'right':
            self.shift_right(tol=tol, nbond=nbond)
        else:
            raise RuntimeError("Failed to shift bond incorrect direction.")

    #updates the MPS so that the site tensors are isometries and return the non-orthogonal bond matrix
    def schmidt_decomposition(self, dir, tol=None, nbond=None, chimin=None):
        if self._orth_centre == None:
            raise RuntimeError("The schmidt decomposition function requires the MPS to be in a mixed canonical form.")
        if (self._orth_centre == 0 and dir  == 'left') or (self._orth_centre + 1 == len(self) and dir == 'right'):
            raise RuntimeError("Unable to perform specified decomposition we are at the bounds of the MPS.")
        
        bind = None
        if dir == 'left':
            bind = self._orth_centre-1
        else:
            bind = self._orth_centre

        il = bind
        ir = bind+1
        
        self._tensors[il], R, self._tensors[ir] = local_canonical_form(self._tensors[il], self._tensors[ir], dir, il, self._orth_centre, tol = tol, nbond = nbond, chimin = chimin)
        return R

    def shift_bond(self, bind, dir='right', tol=None, nbond=None, chimin=None):
        if not self.is_valid():
            raise RuntimeError("The object does not represent a valid MPS.  Unable to perform transformation operations to the MPS.")

        bind = mapint(bind, self.nbonds())

        #get the indices of the two sites that will be modified by the operation
        il = bind
        ir = bind+1

        self._tensors[il], self._tensors[ir] = shift_mps_bond(self._tensors[il], self._tensors[ir], dir, il, self._orth_centre, tol = tol, nbond = nbond, chimin = chimin)
        self._orth_centre = update_mps_ortho_centre(il, ir, self._orth_centre, dir)


    def __str__(self):
        if(self._orth_centre != None):
            return 'ADT: tensors: %s \n orth centre: %d'%(self._tensors, self._orth_centre)
        else:
            return 'ADT: tensors: %s'%(self._tensors)
    
    def __imul__(self, x):
        if(self._orth_centre != None):
            self._tensors[self._orth_centre] *= x
        else:
            self._tensors[0] *= x
        return self

    def __itruediv__(self, x):
        if(self._orth_centre != None):
            self._tensors[self._orth_centre] /= x
        else:
            self._tensors[0] /= x
        return self

    #function for applying a general one-site operator to the MPS
    def apply_one_site(self, M, i):
        if self._is_conjugated:
            M = np.conj(M)
        i = mapint(i, self.nsites())
        
        dM = M.shape[1]
        dims = self._tensors[i].shape
        if(dM != dims[1]):
            raise RuntimeError("The one site operator and MPS site tensor do not have compatible dimensions.")

        if(self._orth_centre != None):
            self.shift_orthogonality(i)
        self._tensors[i] = np.tensordot(M, self._tensors[i], axes=([1], [1]))
        #self._tensors[i] = np.einsum('ij, ajb -> aib', M, self._tensors[i])

    def sparse_prod(M2t):
        M3 = np.zeros((M2t.shape[0], M2t.shape[1], M2t.shape[2], M2t.shape[0], M2t.shape[3]), dtype=np.complex128)
        for i in range(M2t.shape[0]):
            M3[i, :, :, i, :] = M2t[i, :, :, :]
        return M3

    def apply_IF_node(Ik, nt):
        ret = adt.sparse_prod(np.einsum('aj, mjn -> amjn', Ik, nt))
        d = ret.shape
        return (ret.reshape((d[0]*d[1], d[2], d[3]*d[4])))

    def apply_IF_node_end(Ik, nt):
        ret = np.einsum('aj, mjn -> amjn', Ik, nt)
        d = ret.shape
        return (ret.reshape((d[0]*d[1], d[2], d[3])))

    def zipup_left(A, M, tol=None):
        dims = A.shape
        C = adt.apply_IF_node(M, A)
        Cs = C.shape
        #now we swap the tensors around so that up and left point to the right 
        C = C.reshape((Cs[0]*Cs[1], Cs[2]))
        #compute the svd of the c matrix pointing towards the right

        Q, S, Vh = svd(C, full_matrices=False, compute_uv=True, lapack_driver='gesvd')

        nsvd = determine_truncation(S, tol = tol, is_ortho=True)

        S = np.diag(S[:nsvd])
        Vh = Vh[:nsvd, :]

        return Q[:, :nsvd].reshape((Cs[0], Cs[1], nsvd)), S@Vh

    def zipup_internal(A, M, R, tol=None):
        d = A.shape
        B = adt.apply_IF_node(M, A)
        C = np.tensordot(R, B, axes=([1], [0]))
        #C = np.einsum('ij, jkl -> ikl', R, B)
        Cs = C.shape

        C = C.reshape((Cs[0]*Cs[1], Cs[2]))
        Q, S, Vh = svd(C, full_matrices=False, compute_uv=True, lapack_driver='gesvd')
        nsvd = determine_truncation(S, tol = tol, is_ortho=True)

        S = np.diag(S[:nsvd])
        Vh = Vh[:nsvd, :]

        return Q[:, :nsvd].reshape((Cs[0], Cs[1], nsvd)), S@Vh


    def append_first(self, M):
        ms = M.shape
        Mt = M.T
        self._tensors.insert(0, Mt.reshape(1, ms[1], ms[0]))
        self._orth_centre += 1

    def apply_IF_naive(self, M, tol = None, nbond = None):
        for i in range(self.nsites()):
            if(i+1 == len(self)):
                self._tensors[i] = adt.apply_IF_node_end(M[i], self._tensors[i])
            else:
                self._tensors[i] = adt.apply_IF_node(M[i], self._tensors[i])
        self.shift_orthogonality(-1)

    def apply_IF_zipup(self, M, tol=None, nbond=None):
        R = None
        for ind in range(len(self)):
            self._orth_centre = ind
            if(ind == 0):
                self._tensors[ind], R = adt.zipup_left(self._tensors[ind], M[ind], tol=tol)

            elif(ind+1 == len(self)):
                self._orth_centre = ind
                B = adt.apply_IF_node_end(M[ind], self._tensors[ind])
                self._tensors[ind] = np.tensordot(R, B, axes=([1], [0]))
                #self._tensors[ind] = np.einsum('ij, jkl -> ikl', R, B)
            else:
                self._tensors[ind], R = adt.zipup_internal(self._tensors[ind], M[ind], R, tol=tol)


    def apply_IF(self, M, method="naive", tol = None, nbond = None):
        if(len(self) == 1):
            self._tensors[0] = adt.apply_IF_node_end(M[0], self._tensors[0])
            self.orthogonalise()

        else:
            if method == "naive":
                self.apply_IF_naive(M, tol=tol, nbond=nbond)

            elif method == "zipup":
                self.apply_IF_zipup(M, tol=tol, nbond=nbond)

            elif method == "trunc":
                self.apply_IF_trunc(M, tol=tol, nbond=nbond)

            else:
                raise RuntimeError("method not recognised.")

        self.append_first(np.identity(self._D2))
        self.shift_orthogonality(0, tol=tol, nbond=nbond)

    def apply_IF0(self, I0, U):
        if(len(self) > 1):
            raise RuntimeError("Can only apply IF0 to initial density operator.")
        self._tensors[0][0,:, 0]  = U @ (I0*self._tensors[0][0,:, 0])

    def terminate(self):
        M = self._tensors.pop(-1)
        T = np.sum(M, axis=1)
        self._tensors[-1] = np.tensordot(self._tensors[-1], T, axes=([2], [0]))
        #self._tensors[-1] = np.einsum('ijk, kl -> ijl', self._tensors[-1], T)


    def rho(self):
        A = None
        for i in reversed(range(1, len(self))):
            T = np.sum(self._tensors[i], axis=1)
            if not isinstance(A, np.ndarray):
                A = T
            else:
                A = T@A
                #A = np.einsum('ij,jk->ik', T, A)
        rho = None
        if not isinstance(A, np.ndarray):
            rho = self._tensors[0]
        else:
            rho = np.tensordot(self._tensors[0], A, axes=( [2], [0]))
            #rho = np.einsum('ijk, kl -> ijl', self._tensors[0], A)
        return rho[0, :, 0]

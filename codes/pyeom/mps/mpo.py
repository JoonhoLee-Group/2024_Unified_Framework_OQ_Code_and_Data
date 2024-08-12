import numpy as np
from .utils import *


#a basic shared memory mpo implementation.  This contains a number 
class mpo:
    #construct from array of bond dimensions and local hilbert space dimensions
    def __init__(self, chi = None, d = None, N = None, chil : int = 1, chir : int = 1, dtype = np.double, init = 'zeros', do_orthog=True):
        self.dtype = dtype 
        if isinstance(chi, (np.ndarray, list)) and isinstance(d, (np.ndarray, list)):
            self.build_from_arrays(chi, d, chil = chil, chir = chir, dtype = dtype, init = init, do_orthog=do_orthog)
        elif isinstance(chi, int) and isinstance(d, (np.ndarray, list)):
            N = len(d)
            chi_arr = np.ones(N-1, dtype=int)*chi
            self.build_from_arrays(chi_arr, d, chil = chil, chir = chir, dtype = dtype, init = init, do_orthog=do_orthog)
        elif isinstance(chi, (np.ndarray, list)) and isinstance(d, int):
            N = len(chi)+1
            d_arr = np.ones(N, dtype=int)*d
            self.build_from_arrays(chi, d_arr, chil = chil, chir = chir, dtype = dtype, init = init, do_orthog=do_orthog)
        else:
            if N == None:
                self._tensors = None
                self._is_valid = False
                self._boundary_vects = None
                self._orth_centre = None
            else:
                chi_arr = np.ones(N-1, dtype=int)*chi
                d_arr = np.ones(N, dtype=int)*d
                self.build_from_arrays(chi_arr, d_arr, chil = chil, chir = chir, dtype = dtype, init = init, do_orthog=do_orthog)

    def build_from_arrays(self, chi, d, chil = 1, chir = 1, dtype = np.double, init = 'identity', do_orthog=True):
        Nchi = None
        Nd = None
        if not isinstance(chi, (list, np.ndarray)):
            raise TypeError("Bond dimension variable is an invalid type.")
        if not isinstance(d, (list, np.ndarray)):
            raise TypeError("Local Hilbert Space Dimension variable is an invalid type.")

        Nchi = len(chi)
        Nd = len(d)

        if not (Nchi + 1 == Nd):
            raise RuntimeError("bond dimension and local hilbert space arrays are not compatible.")

        self._tensors = [None]*Nd
        if Nd > 1:
            self._tensors[0] = np.zeros((chil, d[0], d[0], chi[0]), dtype=dtype)
            for i in range(1, Nd-1):
                self._tensors[i] = np.zeros((chi[i-1], d[i], d[i], chi[i]), dtype=dtype)
            self._tensors[-1] = np.zeros((chi[-1], d[-1], d[-1], chir), dtype=dtype)
        elif Nd == 1:
            self._tensors[0] = np.zeros((chil, d[0], d[0], chir), dtype=dtype)

        self._orth_centre = None
        self._is_valid = True
        self._boundary_vects = None
        self.init_values(dtype=dtype, init=init)

    def init_values(self, dtype = np.double, init='identity'):
        if isinstance(init, str):
            if init == 'zeros':
                for i in range(len(self)):
                    self._tensors[i] = np.zeros(self._tensors[i].shape, dtype=dtype)
            elif init == 'identity':
                for i in range(len(self)):
                    self._tensors[i] = np.zeros(self._tensors[i].shape, dtype=dtype)
                    self._tensors[i][0, :, :, 0] = np.identity(self._tensors[i].shape[1])

        elif isinstance(init, dict):
            #initialise this to the identity object
            for i in range(len(self)):
                self._tensors[i] = np.zeros(self._tensors[i].shape, dtype=dtype)
                self._tensors[i][0, :, :, 0] = np.identity(self._tensors[i].shape[1])
            if not "op" in init:
                raise ValueError("Invalid dictionary for applying initerator")
            else:
                if "mode" in init:
                    self.apply_one_site(init["op"], init["mode"])
                elif "modes" in init:
                    if len(init["modes"]) == 1:
                        self.apply_one_site(init["op"], init["modes"][0])
                    elif len(init["modes"]) == 2:
                        tol = None
                        nbond = None
                        method = "zipup"
                        if "tol" in init:
                            tol = init["tol"]
                        if "nbond" in init:
                            nbond = init["nbond"]
                        if "method" in init:
                            method = init["method"]
                        self.apply_two_site(init["op"], init["modes"][0], init["modes"][1], method=method, tol=tol, nbond=nbond)
                    else:
                        raise ValueError("Two body or higher initerators only implemented through MPO")
                else:
                    raise ValueError("Failed to read information about modes")



    #access tensor by index
    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[ii] for ii in range(*i.indices(len(self)))]
        elif isinstance(i, int):    
            i = mapint(i, len(self))
            return self._tensors[i]
        else:
            raise TypeError("Invalid argument type")
  
    def __getslice__(self, i):
        return self.__getitem__(i)

    #set tensor by index
    def __setitem__(self, i, v):
        #first we check that v is a valid type
        if not isinstance(v, np.ndarray):
            raise TypeError("Invalid type for setting item.")
        if not v.ndim == 4:
            raise TypeError("Invalid type for setting item.")

        if isinstance(i, slice):
            for ii in range(*i.indices(len(self))):
                self._tensors[ii] = v
        elif isinstance(i, int):    
            i = mapint(i, len(self))
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

    def local_dimension(self, i):
        if isinstance(i, slice):
            return [self.local_dimension(ii) for ii in range(*i.indices(len(self)))]
        elif isinstance(i, int):    
            i = mapint(i, len(self))
            return self._tensors[i].shape[1]
        else:
            raise TypeError("Invalid argument type")

    def expand_local_dimension(self, i, d):
        if isinstance(i, int):    
            i = mapint(i, self.nsites())
            if(d > self._tensors[i].shape[1]):
                npad = d - self._tensors[i].shape[1]
                self._tensors[i] = np.pad(self._tensors[i], ((0,0), (0, npad), (0, npad), (0,0) ), 'constant', constant_values=(0))
        else:
            raise TypeError("Invalid argument type")

    def bond_dimension(self, i):
        if isinstance(i, slice):
            return [self.bond_dimension(ii) for ii in range(*i.indices(len(self)))]
        elif isinstance(i, int):    
            i = mapint(i, self.nbonds())
            return self._tensors[i].shape[3]
        else:
            raise TypeError("Invalid argument type")

    def maximum_bond_dimension(self):
        bd = 0
        for i in range(self.nbonds()):
            if self._tensors[i].shape[3] > bd:
                bd = self._tensors[i].shape[3]
        return bd

    def shape(self, i):
        if isinstance(i, slice):
            return [self.shape(ii) for ii in range(*i.indices(len(self)))]
        elif isinstance(i, int):    
            i = mapint(i, len(self))
            return self._tensors[i].shape
        else:
            raise TypeError("Invalid argument type")

    def expand_bond(self, i, chi):
        if isinstance(i, int):    
            i = mapint(i, self.nbonds())
            if not self._tensors[i].shape[3] == self._tensors[i+1].shape[0]:
                raise RuntimeError("Cannot expand bond, the specified bond is invalid.")
            if(chi > self._tensors[i].shape[3]):
                npad = chi - self._tensors[i].shape[3]
                self._tensors[i] = np.pad(self._tensors[i], ((0,0), (0, 0), (0, 0), (0,npad) ), 'constant', constant_values=(0))
                self._tensors[i+1] = np.pad(self._tensors[i+1], ((0,npad), (0, 0), (0, 0), (0,0) ), 'constant', constant_values=(0))
        else:
            raise TypeError("Invalid argument type")


    #function for ensuring that the mpo is currently valid
    def is_valid(self):
        if not self._is_valid == None:
            return self._is_valid
        else:
            iv = True
            for i in range(len(self._tensors)):
                if not self._tensors[i-1].shape[3] == self._tensors[i].shape[0]:
                    iv = False
            self._is_valid = iv
            return self._is_valid


    def orthogonalise(self):
        for i in reversed(range(self.nbonds())):
            self.shift_bond(i, dir='left')
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


    def shift_bond(self, bind, dir='right', tol=None, nbond=None, chimin=None):
        if not self.is_valid():
            raise RuntimeError("The object does not represent a valid MPS.  Unable to perform transformation operations to the MPS.")

        bind = mapint(bind, self.nbonds())

        #get the indices of the two sites that will be modified by the operation
        il = bind
        ir = bind+1

        
        dl = self._tensors[il].shape
        dr = self._tensors[ir].shape

        Mi, Mj = shift_mps_bond(self._tensors[il].reshape((dl[0], dl[1]*dl[2], dl[3])), self._tensors[ir].reshape((dr[0], dr[1]*dr[2], dr[3])), dir, il, self._orth_centre, tol = tol, nbond = nbond, chimin = chimin)
        self._tensors[il] = Mi.reshape((dl[0], dl[1], dl[2], Mi.shape[-1]))
        self._tensors[ir] = Mj.reshape((Mj.shape[0], dr[1], dr[2], dr[3]))

        self._orth_centre = update_mps_ortho_centre(il, ir, self._orth_centre, dir)


    def expand(self, T, boundary, is_orthogonalised = False):
        if not isinstance(T, np.ndarray):
            raise RuntimeError("Cannot expand mpo as input tensor is invalid.")
        if T.ndim != 4:
            raise RuntimeError("Input tensors is not the correct dimension.")

        if(boundary == 'left'):
            if(T.shape[3] != self._tensors[0].shape[0]):
                raise RuntimeError("Cannot expand MPO boundary tensor has incompatible shape.")
            self._tensors.insert(0, T)
            
            if not is_orthogonalised:
                self._orth_centre = None
            else: 
                self._orth_centre += 1

        elif(boundary == 'right'):
            if(T.shape[0] != self._tensors[self.nbonds()].shape[3]):
                raise RuntimeError("Cannot expand MPO boundary tensor has incompatible shape.")

            self._tensors.append(T)
        else:
            raise RuntimeError("Failed to expand MPO.  Boundary type not recognised")

    def pop(self, boundary):
        if(boundary == 'left'):
            if not self._orth_centre == None:
                if(self._orth_centre > 0):
                    self._orth_centre -= 1
            return self._tensors.pop(0)

        elif(boundary == 'right'):
            if not self._orth_centre == None:
                if(self._orth_centre == self.nbonds() ):
                    self._orth_centre -= 1
            return self._tensors.pop(-1)

        else:
            raise RuntimeError("Failed to pop boundary tensor from MPO.  Boundary type not recognised")

    #function for applying a general one-site operator to the MPO
    def apply_one_site(self, M, i, dir='up', shift_orthogonality = False):
        i = mapint(i, self.nsites())
        
        dM = M.shape[1]
        dims = self._tensors[i].shape
        if(dM != dims[1]):
            raise RuntimeError("The one site operator and MPO site tensor do not have compatible dimensions.")

        if(self._orth_centre != None and shift_orthogonality):
            self.shift_orthogonality(i)

        if(dir == 'up'):
            #self._tensors[i] = np.transpose(np.tensordot(M, self._tensors[i], axes=([1], [1])), axes=(1, 0, 2, 3))
            self._tensors[i] = np.einsum('ij, ajkb -> aikb', M, self._tensors[i])
        elif(dir == 'down'):
            #self._tensors[i] = np.transpose(np.tensordot(M, self._tensors[i], axes=([1], [2])), axes=(1, 2, 0, 3))
            self._tensors[i] = np.einsum('ij, akjb -> akib', M, self._tensors[i])
        else:
            raise ValueError("Failed to apply one site tensor to MPO")

        if(self._orth_centre != i):
            self._orth_centre = None

    def apply_MPO_node(Mt, nt, order = 'mpo', dir='up'):
        ret = None
        if(order == 'mpo'):
            if(dir == 'up'):
                ret = np.einsum('aijb, mjkn -> amikbn', Mt, nt)
            else:
                ret = np.einsum('aijb, mkjn -> amkibn', Mt, nt)
        else:
            if(dir == 'up'):
                ret = np.einsum('aibj, mjkn -> amikbn', Mt, nt)
            else:
                ret = np.einsum('aibj, mkjn -> amkibn', Mt, nt)

        d = ret.shape
        return (ret.reshape((d[0]*d[1], d[2], d[3], d[4]*d[5])))


    def zipup_left(A, M, tol=None, order = 'mpo', dir=dir):
        dims = A.shape
        C = mpo.apply_MPO_node(M, A, order=order, dir=dir)
        Cs = C.shape
        #now we swap the tensors around so that up and left point to the right 
        C = C.reshape((Cs[0]*Cs[1]*Cs[2], Cs[3]))
        #compute the svd of the c matrix pointing towards the right

        Q, S, Vh = np.linalg.svd(C, full_matrices=False, compute_uv=True)

        nsvd = determine_truncation(S, tol = tol, is_ortho=True)

        S = np.diag(S[:nsvd])
        Vh = Vh[:nsvd, :]


        return Q[:, :nsvd].reshape((Cs[0], Cs[1], Cs[2], nsvd)), S@Vh

    def zipup_internal(A, M, R, tol=None, order='mpo', dir=dir):
        d = A.shape
        B = mpo.apply_MPO_node(M, A, order=order, dir=dir)
        C = np.einsum('ij, jklm -> iklm', R, B)
        Cs = C.shape

        C = C.reshape((Cs[0]*Cs[1]*Cs[2], Cs[3]))
        Q, S, Vh = np.linalg.svd(C, full_matrices=False, compute_uv=True)
        nsvd = determine_truncation(S, tol = tol, is_ortho=True)

        S = np.diag(S[:nsvd])
        Vh = Vh[:nsvd, :]

        return Q[:, :nsvd].reshape((Cs[0], Cs[1], Cs[2], nsvd)), S@Vh


    #function for applying a general two-site operator to the MPS
    def apply_two_site(self, M, i, j, method='naive', tol = None, nbond = None, dir='up'):
        i = mapint(i, self.nsites())
        j = mapint(j, self.nsites())

        if(M.shape[0] != M.shape[1]):
            raise RuntimeError("Two site object must be a square matrix.")

        dimsi = self._tensors[i].shape
        dimsj = self._tensors[j].shape

        if(M.shape[0] != dimsi[1]*dimsj[1]):
            raise RuntimeError("The two site object does not have compatible dimensions.")

        if(i == j):
            raise RuntimeError("Cannot apply a two-site operator if only a single unique index has been supplied")
        elif i > j:
            #now shift the ordering of the indices so that i is less than j
            M = np.transpose(M.reshape((dimsi[1], dimsj[1], dimsi[1], dimsj[1])), [1, 0, 3, 2]).reshape((dimsi[1]*dimsj[1], dimsi[1]*dimsj[1]))
            t = i
            i = j
            j = t
            dimsi = self._tensors[i].shape
            dimsj = self._tensors[j].shape

        #now we have set up the state in a suitable form for constructing the 
        if(self._orth_centre == None):
            self.orthogonalise()

        self.shift_orthogonality(i)


        Rt, Vht = two_site_mpo(M, dimsi[1], dimsj[1], order='clockwise', tol=tol)


        if method == "naive":
            self._tensors[i] = mpo.apply_MPO_node(Rt, self._tensors[i], order='clockwise', dir=dir)
            nsvd = Vht.shape[0]
            #now we need to iterate and apply the local MPO objects
            for ind in range(i+1, j):
                d = self._tensors[ind].shape
                self._tensors[ind] = mpo.apply_MPO_node(identity_pad_mpo(nsvd, d[1], order='clockwise'), self._tensors[ind], order='clockwise', dir=dir)
            self._tensors[j] = mpo.apply_MPO_node(Vht, self._tensors[j], order='clockwise', dir=dir)
            self.shift_orthogonality(j)

            #and shift back to the original orthogonality centre truncating all bonds
            self.shift_orthogonality(i, nbond=nbond)

        elif method == "zipup":
            nid = Vht.shape[0]
            self._tensors[i], R = mpo.zipup_left(self._tensors[i], Rt, tol=tol, order='clockwise', dir=dir)

            for ind in range(i+1, j):
                self._orth_centre = ind
                self._tensors[ind], R = mpo.zipup_internal(self._tensors[ind], identity_pad_mpo(nid, self._tensors[ind].shape[1], order='clockwise'), R, tol=tol, order='clockwise', dir=dir)

            self._orth_centre = j
            B = mpo.apply_MPO_node(Vht, self._tensors[j], order='clockwise', dir='dir')
            self._tensors[j] = np.einsum('ij, jklm -> iklm', R, B)
            self.shift_orthogonality(i, tol=tol, nbond=nbond)
        else:
            raise ValueError("Invalid two site mpo mpo contraction scheme.")

    def _apply_nsite(self, M, inds, method='naive', tol = None, nbond = None, optol = None, dir='up'):
        mi = [mapint(x, len(self)) for x in inds]
    
        if len(mi) > len(self):
            raise ValueError("Index array too large.")
        if len(mi) != len(set(mi)):
            raise ValueError("Index array contained duplicates.")

        if(M.shape[0] != M.shape[1]):
            raise ValueError("Input operator is incorrect size.")

        mdim = 1
        dims = [self.local_dimension(m) for m in mi]
        for i, m in enumerate(mi):
            mdim = mdim * self.local_dimension(m)

        if(mdim != M.shape[0]):
            raise ValueError("Input operator and inds array are incompatible for this MPS.")

        #now we have set up the state in a suitable form for constructing the 
        if(self._orth_centre == None):
            self.orthogonalise()

        #now we convert the n-site object into an MPO
        Mt, inds, ds = permute_nsite_dims(M, mi, dims)
        Mp = nsite_mpo(Mt, ds, order='clockwise', tol=optol)


        i = inds[0]
        j = inds[-1]


        c  = 0
        if method == "naive":
            for ind in range(i, j+1):
                print(Mp)
                
                nbd = 1
                if(ind == inds[c]):
                    self._tensors[ind] = mpo.apply_MPO_node(Mp[c], self._tensors[ind], order='clockwise', dir='up')
                    nbd = Mp[c].shape[2]
                    c = c+1
                else:
                    d = self._tensors[ind].shape
                    self._tensors[ind] = mpo.apply_MPO_node(identity_pad_mpo(nbd, d[1], order='clockwise'), self._tensors[ind], order='clockwise', dir='up')

            self.shift_orthogonality(j)

            #and shift back to the original orthogonality centre truncating all bonds
            self.shift_orthogonality(i, tol=tol, nbond=nbond)

        else:
            raise ValueError("Invalid two site mpo mps contraction scheme.")


    #function for applying a general two-site operator to the MPS
    def apply_nsite(self, M, inds, method='naive', tol = None, nbond = None, optol=None, dir='up'):
        if isinstance(inds, int):
            self.apply_one_site(M, inds, dir=dir)
        elif isinstance(inds, list):
            if len(inds) == 1:
                self.apply_one_site(M, inds[0], dir=dir)
            else:
                self._apply_nsite(M, inds, method=method, tol=tol, nbond=nbond, optol=optol, dir=dir)
        else:
            raise RuntimeError("Invalid index object passed to apply_nsite.")



    def apply_MPO_naive(self, M, tol = None, nbond = None, dir='up'):
        check_compatible(self, M, mpo, mpo)
        for i in reversed(range(self.nsites())):
            self.shift_orthogonality(i)
            self._tensors[i] = mpo.apply_MPO_node(M[i], self._tensors[i], dir=dir)

        self.shift_orthogonality(0, tol, nbond)

    def apply_MPO_zipup(self, M, tol=None, nbond=None, dir='up'):
        check_compatible(self, M, mpo, mpo)

        self.shift_orthogonality(0)
        self._tensors[0], R = mpo.zipup_left(self._tensors[0], M[0], tol=tol, dir=dir)

        for ind in range(1, len(self)-1):
            self._orth_centre = ind
            self._tensors[ind], R = mpo.zipup_internal(self._tensors[ind], M[ind], R, tol=tol, dir=dir)

        self._orth_centre = len(self)-1
        B = mpo.apply_MPO_node(M[-1], self._tensors[-1], dir=dir)
        self._tensors[-1] = np.einsum('ij, jklm -> iklm', R, B)
        self.shift_orthogonality(0, tol=tol, nbond=nbond)

    def apply_MPO(self, M, method="naive", tol = None, nbond = None):
        if(len(self) == 1):
            return 
        else:
            if method == "naive":
                self.apply_MPO_naive(M, tol=tol, nbond=nbond)

            elif method == "zipup":
                self.apply_MPO_zipup(M, tol=tol, nbond=nbond)

            else:
                raise RuntimeError("method not recognised.")

    def apply_operator(self, op, method="naive", tol=None, nbond=None):
        if isinstance(op, mpo):
            self.apply_MPO(op, method=method, tol=tol, nbond=nbond)
            return
        elif isinstance(op, dict):
            if not "op" in op:
                raise ValueError("Invalid dictionary for applying operator")
            else:
                if "mode" in op:
                    self.apply_one_site(op["op"], op["mode"])
                elif "modes" in op:
                    if len(op["modes"]) == 1:
                        self.apply_one_site(op["op"], op["modes"][0])
                    elif len(op["modes"]) > 1:
                        if "tol" in op:
                            tol = op["tol"]
                        if "nbond" in op:
                            nbond = op["nbond"]
                        if "method" in op:
                            method = op["method"]
                        self.apply_nsite(op["op"], op["modes"], method=method, tol=tol, nbond=nbond, optol=optol)
                    else:
                        raise ValueError("Two body or higher operators only implemented through MPO")
                else:
                    raise ValueError("Failed to read information about modes")

    def __str__(self):
        if(self._orth_centre != None):
            return 'MPO: tensors: %s \n orth centre: %d'%(self._tensors, self._orth_centre)
        else:
            return 'MPO: tensors: %s'%(self._tensors)
    
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

    def __add__(self, other):
        check_compatible(self, other, mpo, mpo)

        if(self.shape(0)[0] != other.shape(0)[0] or self.shape(-1)[2] != other.shape(-1)[2]):
            raise ValueError("Unable to add mpo objects with different exterior bond dimensions.")
        chil = self.shape(0)[0]
        chir = self.shape(-1)[2]

        bslice = slice(0, len(self)-1)
        chis = [x+y for x,y in zip(self.bond_dimension(bslice), other.bond_dimension(bslice))]
        ds = self.local_dimension(slice(0, len(self)))

        ret = mpo(chi=chis, d=ds, chil=chil, chir=chir, dtype = self.dtype, do_orthog=False)

        cs = self.bond_dimension(0)
        ret[0][:, :, :, 0:cs] = self[0]
        ret[0][:, :, :, cs:chis[0]] = other[0]
        #for all interior tensors we set up the matrices as required 
        for i in range(1, len(self)-1):
            cs = self.bond_dimension(i)
            ret[i][0:cs, :, :, 0:cs] = self[i]
            ret[i][cs:chis[i], :, :, cs:chis[i]] = other[i]

        cs = self.bond_dimension(-1)
        ret[-1][0:cs, :, :, :] = self[-1]
        ret[-1][cs:chis[-1], :, :, :] = other[-1]

        return ret

    def __sub__(self, other):
        check_compatible(self, other, mpo, mpo)

        if(self.shape(0)[0] != other.shape(0)[0] or self.shape(-1)[3] != other.shape(-1)[3]):
            raise ValueError("Unable to add mpo objects with different exterior bond dimensions.")
        chil = self.shape(0)[0]
        chir = self.shape(-1)[3]

        bslice = slice(0, len(self)-1)
        chis = [x+y for x,y in zip(self.bond_dimension(bslice), other.bond_dimension(bslice))]
        ds = self.local_dimension(slice(0, len(self)))

        ret = mpo(chi=chis, d=ds, chil=chil, chir=chir, dtype = self.dtype, do_orthog=False)

        cs = self.bond_dimension(0)
        ret[0][:, :, :, 0:cs] = self[0]
        ret[0][:, :, :, cs:chis[0]] = -other[0]

        #for all interior tensors we set up the matrices as required 
        for i in range(1, len(self)-1):
            cs = self.bond_dimension(i)
            ret[i][0:cs, :, :, 0:cs] = self[i]
            ret[i][cs:chis[i], :, :, cs:chis[i]] = other[i]

        cs = self.bond_dimension(-1)
        ret[-1][0:cs, :, :, :] = self[-1]
        ret[-1][cs:chis[-1], :, :, :] = other[-1]

        return ret




    #def svd_external_bond(self, dir='right', tol = 1e-15):
    #    if not self.is_valid():
    #        raise RuntimeError("The object does not represent a valid MPO.  Unable to perform transformation operations to the MPO.")

    #    #if dir == 'right' we need to reshape and svd the left tensor
    #    if dir == 'right':
    #        il = self.nbonds()
    #        dims = self._tensors[-1].shape
    #        A = self._tensors[il].reshape((dims[0]*dims[1]*dims[2], dims[3]))

    #        Q, S, Vh = np.linalg.svd(A, full_matrices=False, compute_uv=True)

    #        nsvd = len(S)

    #        Q = Q[:, :nsvd]
    #        S = S[:nsvd]
    #        Vh = Vh[:nsvd, :]

    #        Sinv = S
    #        for i in range(nsvd):
    #            if(S[i] > tol):
    #                Sinv[i] = 1.0/S[i]
    #            else:
    #                Sinv[i] = 0
    #        QS = Q @ np.diag(S)
    #        VS = np.diag(S) @ Vh

    #        #check to see if we truncate the bond in any way

    #        self._tensors[il] = QS.reshape((dims[0], dims[1], dims[2], nsvd))
    #        return Sinv, VS


    #    #otherwise we reshape and svd the right tensor
    #    elif dir == 'left':
    #        ir = 0
    #        dims = self._tensors[ir].shape
    #        B = self._tensors[ir].reshape((dims[0], dims[1]*dims[2]*dims[3]))

    #        Q, S, Vh = np.linalg.svd(B, full_matrices=False, compute_uv=True)

    #        nsvd = len(S)

    #        Q = Q[:, :nsvd]
    #        S = S[:nsvd]
    #        Vh = Vh[:nsvd, :]

    #        Sinv = S
    #        for i in range(nsvd):
    #            if(S[i] > tol):
    #                Sinv[i] = 1.0/S[i]
    #            else:
    #                Sinv[i] = 0
    #        QS = Q @ np.diag(S)
    #        VS = np.diag(S) @ Vh

    #        #check to see if we truncate the bond in any way

    #        self._tensors[ir] = VS.reshape( (nsvd, dims[1], dims[2], dims[3]))

    #        return Sinv, QS
    #    else:
    #        ValueError("Invalid dir argument")


    ##function for applying a matrix to the boundary of an MPO.  This is used for inverse canonical gauge based parallelism of larger MPOs 
    #def apply_boundary_matrix(self, M, boundary):
    #    if(boundary == 'left'):
    #        if(M.shape[1] != self._tensors[0].shape[0]):
    #            raise RuntimeError("Cannot apply boundary matrix to MPO.")

    #        #self._tensors[0] = np.tensordot(M, self._tensors[0], axes=([1], [0]))
    #        self._tensors[0] = np.einsum('ij, jabc -> iabc', M, self._tensors[0])
    #        if(self._orth_centre != 0):
    #            self._orth_centre = None

    #    elif(boundary == 'right'):
    #        if(M.shape[0] != self._tensors[self.nbonds()].shape[3]):
    #            raise RuntimeError("Cannot apply boundary matrix to MPO.")

    #        #self._tensors[self.nbonds()] = np.tensordot(self._tensors[self.nbonds()], M, axes=([3],[0]))
    #        self._tensors[self.nbonds()] = np.einsum('ij, cabi -> cabj', M, self._tensors[self.nbonds()])
    #        if(self._orth_centre != self.nbonds()):
    #            self._orth_centre = None

    #    else:
    #        raise RuntimeError("Failed to apply boundary matrix.  Boundary type not recognised")


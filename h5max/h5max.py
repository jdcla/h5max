import numpy as np
import h5py
from scipy import sparse

def store_sparse_matrix(M, fh):
    """
    Store a csr matrix in HDF5 (based on h5py syntax)

    Parameters
    ----------
    M: scipy.sparse
        sparse matrix
    fh: str
        handle to destination HDF5 group 
    """
    assert(M.__class__ == sparse.csr.csr_matrix), 'M must be a csr matrix'
    for attribute in ('data', 'indices', 'indptr', 'shape'):
        # remove existing nodes
        if attribute in fh.keys():  
            del fh[attribute]
        # add nodes
        arr = np.array(getattr(M, attribute))
        fh.create_dataset(attribute, data=arr)


def load_sparse_matrix(fh):
    """
    Load a csr matrix from HDF5 (based on h5py syntax)

    Parameters
    ----------
    fh: str
        handle to destination HDF5 group 

    Returns
    ----------
    M : scipy.sparse.csr.csr_matrix
        loaded sparse matrix
    """

    # get nodes
    attributes = []
    for attribute in ('data', 'indices', 'indptr', 'shape'):
        attributes.append(fh[attribute])

    # construct sparse matrix
    M = sparse.csr_matrix(tuple(attributes[:3]), shape=attributes[3])
    return M


def store_sparse_matrices(Ms, fh):
    """
    Store a list of matrices in HDF5 (based on h5py syntax). Attributes of a single
    matrix are stored at the same index for the different attribute datasets

    Parameters
    ----------
    Ms: List(scipy.sparse)
        list of sparse matrices
    fh: str
        handle to destination HDF5 group 
    """
    data = {'data':[], 'indices':[], 'indptr':[], 'shape':[]}
    for sample in Ms:
        for attribute in data.keys():
            data[attribute].append(np.array(getattr(sample, attribute)))
    
    for attribute in data.keys():
        # remove existing nodes
        if attribute in fh.keys():  
            del fh[attribute]
        att_dtype = data[attribute][0].dtype
        att_lens = np.array([len(d) for d in data[attribute]])
        if (att_lens[0] == att_lens).all():
            fh.create_dataset(attribute, data=data[attribute])
        else:
            fh.create_dataset(attribute, data=data[attribute],
                              dtype=h5py.vlen_dtype(att_dtype))
            
            
def load_sparse_matrices(ids, fh):
    """
    load a list of csr matrix in HDF5 (based on h5py syntax)

    Parameters
    ----------
    ids: List(int)
        list of indices
    fh: str
        handle to source HDF5 group 
    """
    data = []
    for idx in ids:
        attributes = []
        for attribute in ('data', 'indices', 'indptr', 'shape'):
            attributes.append(fh[attribute][idx])
        # construct sparse matrix
        data.append(sparse.csr_matrix(tuple(attributes[:3]), shape=attributes[3]))
    return data
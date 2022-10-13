import numpy as np
import h5py
from scipy import sparse

format_dict = {'csc': sparse.csc_matrix,
               'csr': sparse.csr_matrix,
#               'coo': sparse.coo_matrix,
#               'bsc': sparse.bsr_matrix,
#               'dia': sparse.dia_matrix,
#               'dok': sparse.dok_matrix,
#               'lil': sparse.lil_matrix,
               }

format_attr_dict = {'csc': ['data', 'indices', 'indptr', 'shape'],
                    'csr': ['data', 'indices', 'indptr', 'shape'],
                    }

def store_sparse_matrices(fh, Ms, format='csr', overwrite=False):
    """
    Store a list of matrices in HDF5 (based on h5py syntax). Attributes of a single
    matrix are stored at the same index for the different attribute datasets. Matrices are
    expected to be sparse.

    Parameters
    ----------
    Ms: np.array OR list(np.array)
        list of np.array matrices
    fh: str
        handle to destination HDF5 group
    format:
        sparse storing strategy utilized by scipy.
        supported types are [csc, csr]
    overwrite: bool
        whether to overwrite existing nodes by default or raise an error
    """
    if type(Ms) == np.array:
        Ms = [Ms]
    data = {key: [] for key in format_attr_dict[format]}
    
    for sample in Ms:
        sample_s = format_dict[format](sample)
        for attribute in data.keys():
            data[attribute].append(np.array(getattr(sample_s, attribute)))
    
    for attribute in data.keys():
        # remove existing nodes
        if overwrite and attribute in fh.keys():  
            del fh[attribute]
        att_dtype = data[attribute][0].dtype
        att_lens = np.array([len(d) for d in data[attribute]])
        try:
            if (att_lens[0] == att_lens).all():
                fh.create_dataset(attribute, data=data[attribute])
            else:
                fh.create_dataset(attribute, data=data[attribute], dtype=h5py.vlen_dtype(att_dtype))
        except ValueError as e:
            p = e.args[0].split('dataset')
            e.args = (p[0] + "dataset \"%s\"" % attribute + p[-1] +
            ". Did you mean to specify `overwrite=True`?", )
            raise 
                
            
            
            
def load_sparse_matrices(fh, idxs=None, format='csr'):
    """
    load a list of sparse matrices from a HDF5 group

    Parameters
    ----------
    fh: str
        handle to source HDF5 group
    idxs: [int, list(int), None]
        single index or list of indices. If no indexes are given,
        all matrices are loaded.
    format:
        sparse storing strategy utilized by scipy.
        supported types are [csc, csr]
        
    Returns
    ----------
    Ms : np.array OR list(np.array)
    """
    if type(idxs) == int:
        return load_sparse_matrix(fh, idxs, format=format)
    data = []
    if idxs is None:
        idxs = np.arange(len(fh[format_attr_dict[format][0]]))
    for idx in idxs:
        data.append(load_sparse_matrix(fh, idx, format=format))
        
    return data

def load_sparse_matrix(fh, idx, format):
    """
    load a single sparse matrix from a HDF5 group

    Parameters
    ----------
    fh: str
        handle to source HDF5 group
    idx: [int, list(int), None]
        single index or list of indices. If no indexes are given,
        all matrices are loaded.
    format:
        sparse storing strategy utilized by scipy.
        supported types are [csc, csr]
        
    Returns
    ----------
    M : np.array
    """
    attributes = []
    for attribute in format_attr_dict[format]:
        attributes.append(fh[attribute][idx])
    # construct sparse matrix
    M = format_dict[format](tuple(attributes[:3]), shape=attributes[3]).toarray()
    
    return M
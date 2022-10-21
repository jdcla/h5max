import numpy as np
import h5py
from scipy import sparse
from typing import Optional, Literal, Union, List, TypeVar

format_dict = {
    'csr': sparse.csr_matrix,
    'csc': sparse.csc_matrix,
    'coo': sparse.coo_matrix,
#   'bsr': sparse.bsr_matrix,   <- could be implemented, need extra attribute to describe data shape
#   'dia': sparse.dia_matrix,   <- could be implemented, need extra attribute to describe data shape
#   'dok': sparse.dok_matrix,   <- seems not feasible
#   'lil': sparse.lil_matrix,   <- seems not feasible
}

S = TypeVar("S", *list(format_dict.keys()))

format_attr_dict = {
    'csr': ['data', 'indices', 'indptr', 'shape'],
    'csc': ['data', 'indices', 'indptr', 'shape'],
    'coo': ['data', 'row', 'col', 'shape'],
    'bsr': ['data', 'indices', 'indptr', 'shape'],
    'dia': ['data', 'offsets', 'shape'],
}

def store_sparse_matrices(
    f: Union[h5py._hl.group.Group, h5py._hl.files.File],
    data: Union[np.ndarray, List[np.ndarray], S, List[S]], 
    format: Literal["csc", "csr", "coo"] = "csr", 
    overwrite: bool = False,
    ):
    """Store a list of matrices in HDF5 (based on h5py syntax). Attributes of a single
    matrix are stored at the same index for the different attribute datasets. Matrices are
    expected to be already sparse or numpy arrays.

    Args:
        f (Union[h5py._hl.group.Group, h5py._hl.files.File]): handle to destination HDF5 group
        data (Union[np.ndarray, List[np.ndarray], S, List[S]]): matrix or list of matrices
        format (Literal[&quot;csc&quot;, &quot;csr&quot;], optional): sparse storing strategy 
            utilized by scipy. Ignored when a sparse matrix or a list of sparse matrices is given for
            data. Defaults to "csr".
        overwrite (bool, optional): whether to overwrite existing nodes by default or raise an error.
            Defaults to False.
    """
    if type(data) != list:
        data = [data]
    transform = type(data[0]) == np.ndarray
    data_attr = {key: [] for key in format_attr_dict[format]}
    
    for sample in data:
        if transform:
            sample = format_dict[format](sample)
        for attribute in data_attr.keys():
            data_attr[attribute].append(np.array(getattr(sample, attribute)))
    
    for attribute in data_attr.keys():
        # remove existing nodes
        if overwrite and attribute in f.keys():  
            del f[attribute]
        att_dtype = data_attr[attribute][0].dtype
        att_lens = np.array([len(d) for d in data_attr[attribute]])
        try:
            if (att_lens[0] == att_lens).all():
                f.create_dataset(attribute, data=data_attr[attribute])
            else:
                f.create_dataset(attribute, data=data_attr[attribute], dtype=h5py.vlen_dtype(att_dtype))
        except ValueError as e:
            p = e.args[0].split('dataset')
            e.args = (p[0] + "dataset \"%s\"" % attribute + p[-1] +
            ". Did you mean to specify `overwrite=True`?", )
            raise        
            
def load_sparse_matrices(
    f: Union[h5py._hl.group.Group, h5py._hl.files.File],
    idxs: Union[int, List[int], None],
    format: Literal["csr", "csc", "coo"] = "csr",
    to_numpy: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
    """load a list of sparse matrices from a HDF5 group

    Args:
        f (Union[h5py._hl.group.Group, h5py._hl.files.File]): handle to source HDF5 group
        idxs (Union[int, List[int], None]):single index or list of indices. If no indexes are given, 
        all matrices are loaded.
        format (Literal[&quot;csr&quot;, &quot;csc&quot;, &quot;coo&quot;], optional): sparse storing 
        strategy utilized by scipy. Defaults to "csr".
        to_numpy (bool), optional: return dense numpy array. Defaults to True.

    Returns:
        Union[np.ndarray, List[np.ndarray]: matrix or list of matrices
    """
    if type(idxs) == int:
        return load_sparse_matrix(f, idxs, format=format, to_numpy=to_numpy)
    data = []
    if idxs is None:
        idxs = np.arange(len(f[format_attr_dict[format][0]]))
    for idx in idxs:
        data.append(load_sparse_matrix(f, idx, format=format, to_numpy=to_numpy))
        
    return data

def load_sparse_matrix(
    f: Union[h5py._hl.group.Group, h5py._hl.files.File],
    idx: int,
    format: Literal["csr", "csc", "coo"] = "csr",
    to_numpy: bool = True
    ) -> np.ndarray:
    """load a single sparse matrix from a HDF5 group

    Args:
        f (Union[h5py._hl.group.Group, h5py._hl.files.File]): handle to source HDF5 group
        idx (int): single index 
        format (Literal[&quot;csr&quot;, &quot;csc&quot;, &quot;coo&quot;], optional): 
        sparse storing strategy utilized by scipy. Defaults to "csr".
        to_numpy (bool), optional: return dense numpy array. Defaults to True.

    Returns:
        np.ndarray
    """
    attributes = []
    for attribute in format_attr_dict[format]:
        attributes.append(f[attribute][idx])
    # construct sparse matrix
    array = format_dict[format](tuple(attributes[:-1]), shape=attributes[-1])
    if to_numpy:
        array = array.toarray()
    
    return array
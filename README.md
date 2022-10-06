# h5max

A utility package built upon `h5py` for easier data saving and loading.

This package features:
- facilitated saving and loading of sparse matrices using `scipy`.

## Installation

```
pip install h5max
```

## Usage

```
import h5py
import h5max
import numpy as np

fh = h5py.File('my_data.h5', 'w')

a = np.zeros((100,100))
b = np.zeros((1000,50))
a[7,1] = 1
b[1,0] = 10

Ms = [a, b]

# store both a, b
h5max.store_sparse_matrices(fh, Ms, format='csr')

# load only a (index 0)
a_out = h5max.load_sparse_matrices(fh, 0, format='csr')

# load [a,b]
Ms_out = h5max.load_sparse_matrices(fh, [0, 1], format='csr')

fh.close()
```


![h5max_pic](https://github.com/jdcla/h5max/blob/main/h5max.png)

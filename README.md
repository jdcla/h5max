<div align="center">
<h1>h5max</h1>

A utility package built upon `h5py` for easier data saving and loading of sparse data objects.

[![PyPi Version](https://img.shields.io/pypi/v/h5max.svg)](https://pypi.python.org/pypi/h5max/)
[![GitHub license](https://img.shields.io/github/license/jdcla/h5max)](https://github.com/jdcla/h5max/blob/main/LICENSE.md)
[![GitHub issues](https://img.shields.io/github/issues/jdcla/h5max)](https://github.com/jdcla/h5max/issues)
[![GitHub stars](https://img.shields.io/github/stars/jdcla/h5max)](https://github.com/jdcla/h5max/stargazers)
</div>

`h5max` handles storing and loading of `scipy.sparse` data structures in `h5py` file objects, which is not natively supported. It assumes a simple data structure where information of individual samples are stored according to the index they occupy within datasets.  

<div align="center">
<img src="https://github.com/jdcla/h5max/raw/main/h5max.png" width="600">
</div>

## 🔗 Installation

```bash
pip install h5max
```

## 📖 User guide

```python
import h5py
import h5max
import numpy as np

fh = h5py.File('my_data.h5', 'w')

a = np.zeros((100,100))
b = np.zeros((1000,50))
a[7,1] = 1
b[1,0] = 10

m_list = [a, b]

# store both a, b
h5max.store_sparse(fh, m_list, format='csr')

# load only a (index 0)
a_out = h5max.load_sparse(fh, 0, format='csr')

# load [a,b]
m_list_out = h5max.load_sparse(fh, [0, 1], format='csr', to_numpy=True)

# load all idxs in the data
m_list_out = h5max.load_sparse(fh, format='csr')

fh.close()
```

## ✔️ Package features

- [x] Support for `csr`, `csc`, `coo` sparse types
- [ ] Support for `bsr`, `dia`, `dok`, `lil` sparse types
- [x] Support for overwriting
- [x] Flexible data loading and saving (both as sparse and numpy arrays.)
- [ ] Automatic format detection

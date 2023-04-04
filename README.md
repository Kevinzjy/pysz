# pysz
Python interface for writing and reading svb-zstd compressed data

## Installation

### Dependencies

- Python >= 3.8
- Numpy >= 1.24.2
- Pandas >= 2.0.0
- Zstandard >= 0.20.0

### Install using pip

```bash
# First, install a customized version of pystreamvbyte
pip install git+https://github.com/kevinzjy/pystreamvbyte.git

# Install pysz now
pip install pysz
```

### Install from source

```bash
# First, install a customized version of pystreamvbyte
pip install git+https://github.com/kevinzjy/pystreamvbyte.git

# Install pysz from source
git clone --recursive https://github.com/kevinzjy/pysz
cd pysz
pip install .
```

## Usage

### Create & Write SZ file 

Supported data types：

- str
- np.int16 / np.int32
- np.uint16 / np.uint32
- np.float16 / np.float32 / np.float64 / np.float128

```python
import numpy as np
from pysz.api import CompressedFile

header = [('version',  '0.0.1'), ('date', '2023-04-03')]
attr = [('ID', str), ('Offset', np.int32), ('Raw_unit', np.float32)]
datasets = [('Raw', np.uint32), ('Fastq', str), ('Move', np.uint16)]

# Create new SZ file
sz = CompressedFile(
    "/tmp/test_sz", mode="w",
    header=header, attributes=attr, datasets=datasets,
    overwrite=True, n_threads=8
)

# Save data in single-read mode 
for i in range(10000):
    sz.put(
        f"read_{i}", # ID
        0, # Offset
        np.random.rand(), # Raw_unit
        np.random.randint(70, 150, 4000), # Raw
        ''.join(np.random.choice(['A', 'T', 'C', 'G'], 450)), # Fastq
        np.random.randint(0, 1, 4000), # Move
    )

# Save data in chunk mode
for i in range(100):
    chunk = []
    for j in range(100):
        chunk.append((
            f"read_{i}_{j}",
            0,
            np.random.rand(),
            np.random.randint(70, 150, 4000),
            ''.join(np.random.choice(['A', 'T', 'C', 'G'], 450)),
            np.random.randint(0, 1, 4000),
        ))
    sz.put_chunk(chunk)

# Remember to call it to ensure everything finished successfully
sz.close()
```

Examples for reading SZ file

```python
import numpy as np
from pysz.api import CompressedFile
sz = CompressedFile(
        "/tmp/test_sz", mode="r", allow_multiprocessing=True,
)

# Get index information
print(sz.idx.head())

# Get total read number
print(f"Loaded {sz.idx.shape[0]} reads")

# Get the first read
read = sz.get(0)
print(read.ID, read.Offset, read.Raw_unit, read.Raw, read.Fastq, read.Move)

# Get first 10 reads
reads = sz.get(np.arange(10))

# Filter some reads
idx = sz.idx.index[sz.idx['ID'].isin(['read_0', 'read_1'])]
reads = sz.get(idx)
```

Example for reading SZ file chunk by chunk using multiprocessing

> Note: for reading SZ file with multiprocessing, pass the chunked index as parameter, 
> and init CompressedFile instance with `allow_multiprocessing=True` separately in each process.

```python
import numpy as np
from pysz.api import CompressedFile
from pysz.utils import error_callback
from multiprocessing import Pool

def grouper(iterable, n, fillvalue=None):
    """
    Collect data info fixed-length chunks or blocks
    grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    """
    from itertools import zip_longest
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def read_from_sz(file_name, idx):
    """
    Read from SZ inside each process
    """
    sz = CompressedFile(file_name, mode='r', allow_multiprocessing=True)
    reads = sz.get(idx)
    sz.close()
    return reads
    
# Init pool
n_threads = 16
chunk_size = 500

# Get all reads stored in SZ file
file_name = '/tmp/test_sz'
sz = CompressedFile(file_name, mode='r')
index = sz.idx
sz.close()

# Add process to Pool
pool = Pool(n_threads)
jobs = []
for x in grouper(index.index, chunk_size):
    chunk = [i for i in x if i is not None]
    jobs.append(pool.apply_async(read_from_sz, (file_name, chunk, ), error_callback=error_callback))
pool.close()

# Get output
for job in jobs:
    ret = job.get()
    # Process your data here
pool.join()
```

## TODO

- Add support for zfp for lossy float compression
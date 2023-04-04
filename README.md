# pysz
Python interface for writing and reading svb-zstd compressed data

## Installation

```bash
# First, install a customized version of pystreamvbyte
pip install git+https://github.com/kevinzjy/pystreamvbyte.git

# Install pysz now
pip install pysz
```

## Usage

Examples for creating new SZ file and saving some data

```python
import numpy as np
from pysz.api import CompressedFile

header = [('version',  '0.0.1'), ('date', '2023-04-03')]
attr = [('ID', str), ('Offset', np.int32), ('Raw_unit', np.float32)]
datasets = [('Raw', np.uint32), ('Fastq', str), ('Move', np.uint16), ('Norm', np.uint32)]

# Create new SZ file
sz = CompressedFile(
    "/tmp/test_sz", mode="w",
    header=header, attributes=attr, datasets=datasets,
    overwrite=True, n_threads=8
)

# Save some reads
for i in range(100):
    sz.put(
        f"read_{i}", #ID
        0, #Offset
        np.random.rand(), #Raw_unit
        np.random.randint(70, 150, 4000), #Raw
        ''.join(np.random.choice(['A', 'T', 'C', 'G'], 450)), #Fastq
        np.random.randint(0, 1, 4000), #Move
    )

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

# Get the first read
read = sz.get(0)
print(read.ID, read.Offset, read.Raw_unit, read.Raw, read.Fastq, read.Move)

# Get first 10 reads
reads = sz.get(np.arange(10))

# Filter some reads
idx = sz.idx.index[sz.idx['ID'].isin(['read_0', 'read_1'])]
reads = sz.get(idx)
```
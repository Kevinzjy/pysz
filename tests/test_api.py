import pytest
import numpy as np
from pathlib import Path
from datetime import datetime
from pysz import __version__
from pysz.api import CompressedFile

class TestInit:

    def test_version(self):
        print("version", __version__)

class TestCompressedFile:

    @pytest.fixture
    def test_write(self):
        dir_name = Path() / "test_sz"

        header = [('version',  __version__), ('args', 'null')]
        attr = [('ID', str), ('Offset', np.int32), ('Raw_unit', np.float32)]
        datasets = [('Raw', np.uint16), ('Fastq', str), ('Move', np.uint16), ('Scores', np.float32)]

        sz = CompressedFile(
            dir_name, mode="w",
            header=header, attributes=attr, datasets=datasets,
            overwrite=True, n_threads=4
        )

        st = datetime.now()
        cnter = 0
        for i in range(100):
            sz.put(
                f"read_{cnter}",
                0,
                np.random.rand(),
                np.random.randint(70, 150, 4000),
                ''.join(np.random.choice(['A', 'T', 'C', 'G'], 450)),
                np.random.randint(0, 1, 4000),
                np.random.random(4000),
            )
            cnter += 1
        print(f"Saved 10000 reads in single-read mode in {dir_name} using {datetime.now()-st}s")

        st = datetime.now()
        for _ in range(10):
            chunk = []
            for i in range(10):
                chunk.append((
                    f"read_{cnter}",
                    0,
                    np.random.rand(),
                    np.random.randint(70, 150, 4000),
                    ''.join(np.random.choice(['A', 'T', 'C', 'G'], 450)),
                    np.random.randint(0, 1, 4000),
                    np.random.random(4000),
                ))
                cnter += 1
            sz.put_chunk(chunk)
        print(f"Saved 10000 reads in chunk mode in {dir_name} using {datetime.now()-st}s")

        sz.close()
        yield dir_name

        sz.idx_path.unlink()
        sz.dat_path.unlink()
        dir_name.rmdir()

    def test_read(self, test_write):
        sz = CompressedFile(
            test_write, mode="r", allow_multiprocessing=True
        )
        print(f"Loaded {sz.idx.shape[0]} reads")

        reads = sz.get(sz.idx.sample(n=100).index)
        _ = ','.join([i.ID for i in reads])
        print(f"Successively sampled and parsed 100 reads")

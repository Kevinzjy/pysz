import pytest
import numpy as np
from pathlib import Path
from pysz import __version__
from pysz.api import CompressedFile

class TestInit():
    def test_version(self):
        print("version", __version__)


class TestCompressedFile():

    @pytest.fixture
    def test_write(self):
        dir_name = Path(__file__).parent / "test_sz"

        header = [('version',  __version__), ('args', 'null')]
        attr = [('ID', str), ('Offset', np.int32), ('Raw_unit', np.float32)]
        datasets = [('Raw', np.uint32), ('Fastq', str), ('Move', np.uint16), ('Norm', np.uint32)]

        sz = CompressedFile(
            dir_name, mode="w",
            header=header, attributes=attr, datasets=datasets,
            overwrite=True, n_threads=4
        )

        for i in range(100):
            sz.put(
                f"read_{i}",
                0,
                np.random.rand(),
                np.random.randint(70, 150, 4000),
                ''.join(np.random.choice(['A', 'T', 'C', 'G'], 450)),
                np.random.randint(0, 1, 4000),
                np.random.randint(70, 150, 4000),
            )
        print(f"Saved 100 reads in {dir_name}")

        sz.close()
        yield dir_name

        sz.idx_path.unlink()
        sz.dat_path.unlink()
        dir_name.rmdir()

    def test_read(self, test_write):
        sz = CompressedFile(
            test_write, mode="r", allow_multiprocessing=True
        )
        reads = sz.get([0, 1, 2, 3, 4, 5])
        read_ids = ','.join([i.ID for i in reads])
        print(f"Successively retrived read: {read_ids}")

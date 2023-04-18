import pytest
import numpy as np
from pathlib import Path
from datetime import datetime
from pysz import __version__
from pysz.api import CompressedFile
from multiprocessing import Pool

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

        # st = datetime.now()
        for i in range(100):
            sz.put(
                f"read_{i}",
                0,
                np.random.rand(),
                np.random.randint(70, 150, 4000),
                ''.join(np.random.choice(['A', 'T', 'C', 'G'], 450)),
                np.random.randint(0, 1, 4000),
                np.random.random(4000),
            )
        # print(f"Saved 10000 reads in single-read mode in {dir_name} using {datetime.now()-st}s")

        # st = datetime.now()
        for i in range(10):
            chunk = []
            for j in range(10):
                chunk.append((
                    f"chunk{i}_{j}",
                    0,
                    np.random.rand(),
                    np.random.randint(70, 150, 4000),
                    ''.join(np.random.choice(['A', 'T', 'C', 'G'], 450)),
                    np.random.randint(0, 1, 4000),
                    np.random.random(4000),
                ))
            sz.put_chunk(chunk)
        # print(f"Saved 10000 reads in chunk mode in {dir_name} using {datetime.now()-st}s")

        sz.close()
        yield dir_name

        sz.idx_path.unlink()
        sz.dat_path.unlink()
        dir_name.rmdir()

    def test_read(self, test_write):
        sz = CompressedFile(
            test_write, mode="r", allow_multiprocessing=True
        )
        assert sz.idx.shape[0] == 200

        reads = sz.get(sz.idx.index)
        n_read = sum([i.ID.startswith("read") for i in reads])
        n_chunk = sum([i.ID.startswith("chunk") for i in reads])
        assert n_read == 100
        assert n_chunk == 100

    @staticmethod
    def writer(queue, chunk_id):
        for i in range(10):
            read = (
                f"read_{chunk_id}_{i}",
                0,
                np.random.rand(),
                np.random.randint(70, 150, 4000),
                ''.join(np.random.choice(['A', 'T', 'C', 'G'], 450)),
                np.random.randint(0, 1, 4000),
                np.random.random(4000),
            )
            queue.put((False, read))

        chunk = []
        for i in range(10):
            chunk.append((
                f"chunk_{chunk_id}_{i}",
                0,
                np.random.rand(),
                np.random.randint(70, 150, 4000),
                ''.join(np.random.choice(['A', 'T', 'C', 'G'], 450)),
                np.random.randint(0, 1, 4000),
                np.random.random(4000),
            ))
        queue.put((True, chunk))

    @pytest.fixture
    def test_parallel_write(self):
        dir_name = Path() / "test_sz_parallel"

        header = [('version',  __version__), ('args', 'null')]
        attr = [('ID', str), ('Offset', np.int32), ('Raw_unit', np.float32)]
        datasets = [('Raw', np.uint16), ('Fastq', str), ('Move', np.uint16), ('Scores', np.float32)]

        sz = CompressedFile(
            dir_name, mode="w",
            header=header, attributes=attr, datasets=datasets,
            overwrite=True, n_threads=4
        )

        st = datetime.now()
        pool = Pool(8)
        jobs = []
        for i in range(10):
            jobs.append(pool.apply_async(self.writer, (sz.q_in, i, )))
        pool.close()
        pool.join()

        # print(f"Saved 1000 reads in single-read mode in {dir_name} using {datetime.now()-st}s")
        sz.close()
        yield dir_name

        sz.idx_path.unlink()
        sz.dat_path.unlink()
        dir_name.rmdir()

    def test_parallel_read(self, test_parallel_write):
        sz = CompressedFile(
            test_parallel_write, mode="r", allow_multiprocessing=True
        )
        assert sz.idx.shape[0] == 200

        reads = sz.get(sz.idx.index)
        n_read = sum([i.ID.startswith("read") for i in reads])
        n_chunk = sum([i.ID.startswith("chunk") for i in reads])
        assert n_read == 100
        assert n_chunk == 100

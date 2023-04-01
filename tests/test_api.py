import pytest
import numpy as np
from pathlib import Path
from pysz import __version__
from pysz.api import CompressedFile

class TestModule():
    def test_version(self):
        print(__version__)


class TestCompressedFile():
    def test_file(self):
        compressed_file = CompressedFile(Path()/"test_sz", "w", overwrite=True)

        header = [('version',  __version__), ('args', 'null')]
        attr = [('ID', str), ('Offset', np.int32), ('Raw_unit', np.float32)]
        datasets = [('Raw', np.uint32), ('Fastq', str), ('Move', np.uint16), ('Norm', np.uint32)]

        compressed_file.write_template(header, attr, datasets)

        encoded_line, idx_line = compressed_file.write_record(
            "read1",
            1,
            1.14,
            np.arange(100),
            "AGCTAGTCGTACT" * 10,
            np.zeros(100),
            np.arange(100) * 2,
        )
        print(encoded_line)
        print(idx_line)



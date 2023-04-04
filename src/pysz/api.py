import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict, namedtuple
from zstandard import ZstdDecompressor

from pysz import __version__
from pysz.compression import svb_decode, svb_encode, zstd_decode, zstd_encode, str_decode, str_encode
from pysz.utils import mkdir, assert_file_exists, assert_dir_exists

from multiprocessing import Manager, Process

# Default attributes & datasets for chunk data
Attributes = [
    ('ID', str),
    ('Offset', np.int32),
    ('Raw_unit', np.uint32),
]
Datasets = [
    ('Raw', np.uint32),
    ('Fastq', str),
    ('Move', np.uint16),
]

# For dtype conversion
Dtypes_names = {
    str: "S",
    np.int16: "I16", np.int32: "I32",
    np.uint16: "U16", np.uint32: "U32",
    np.float16: "F16", np.float32: "F32",
}
Dtypes = {j: i for i, j in Dtypes_names.items()}

# Dtypes that can be svb compressed
Svb_dtypes = [np.int16, np.int32, np.uint16, np.uint32]


class CompressedFile(object):
    """
    Class for parsing SVB-ZSTD compressed data
    """

    def __init__(self, data_dir, mode:str="r", header:list=None, attributes:list=None, datasets:list=None, overwrite:bool=False, n_threads=8, allow_multiprocessing=True):
        """
        Init function of CompressedFile Class

        Args:
            data_dir: str / Path,
                path to SZ directory, required
            mode: str,
                "r" for read and "w" for writing sz data
            header: [(key, value)]:
                list of of keys and values in header, defaults to [(version, '0.0.1')]
            attributes: [(attr_id, attr_dtype)],
                list of ID and dtype for attributes, defaults to [('ID', str), ('Offset', np.int32), ('Raw_unit', np.uint32)]
            datasets: [(dataset_id, dataset_dtype)],
                list of ID and dtype for datasets, defaults to [('Raw', np.uint32), ('Fastq', np.uint32), ('Move', np.uint32)]
            overwrite: bool,
                whether overwrite existed sz data, defaults to False
            n_threads: int,
                number of threads for parallelized data compression, defaults to 8
            allow_multiprocessing: bool,
                set to True if want to access SZ data using multiprocessing, defaults to False
        """
        self.dir = data_dir if isinstance(data_dir, Path) else Path(data_dir)
        self.idx_path = self.dir / "index"
        self.dat_path = self.dir / "dat"
        self.mode = mode

        # For Record class
        self.idx, self.header, self.attr, self.datasets = None, None, None, None

        # For multiprocessing
        self.threads = n_threads
        self.q_in = Manager().Queue()
        self.q_out = Manager().Queue()
        self.pool = []
        self.worker, self.decompressor, self.fh = None, None, None

        # Get attributes and datasets
        if self.mode == "r":
            assert_dir_exists(data_dir)
            assert_file_exists(self.idx_path)
            assert_file_exists(self.dat_path)
            self.load_header()
        elif self.mode == 'w':
            mkdir(data_dir, overwrite=overwrite)
            self.save_header(header, attributes, datasets)
        else:
            raise KeyError("Mode should be either 'w' or 'r'.")

        # Init Record class
        keys = []
        if self.attr is not None:
            keys += list(self.attr)
        if self.datasets is not None:
            keys += list(self.datasets)
        self.record = namedtuple("Record", keys)

        # Init multiprocessing related variables
        if self.mode == 'w':
            self.worker = Process(target=self.writer)
            self.worker.start()
            for _ in np.arange(self.threads):
                p = Process(target=self.encoder)
                p.start()
                self.pool.append(p)
        else:
            if allow_multiprocessing:
                self.decompressor = ZstdDecompressor()
                self.fh = open(self.dat_path, 'rb')

    def save_header(self, header, attributes, datasets):
        """
        Save header, attributes and dataset information to the index file

        Args:
            header: [(key, value)]:
                list of of keys and values in header, defaults to [(version, '0.0.1')]
            attributes: [(attr_id, attr_dtype)],
                list of ID and dtype for attributes, defaults to [('ID', str), ('Offset', np.int32), ('Raw_unit', np.uint32)]
            datasets: [(dataset_id, dataset_dtype)],
                list of ID and dtype for datasets, defaults to [('Raw', np.uint32), ('Fastq', np.uint32), ('Move', np.uint32)]
        """
        self.header =  {"version": __version__} if header is None else dict(header)
        self.attr =  OrderedDict(Attributes) if attributes is None else OrderedDict(attributes)
        self.datasets = OrderedDict(Datasets) if datasets is None else OrderedDict(datasets)

        with open(self.idx_path, 'w') as out:
            header_str = json.dumps(self.header)
            out.write("#" + header_str + '\n')

            cols = ["LENGTH", "OFFSET",] + \
                [f"{i}:A:{Dtypes_names[j]}" for i,j in self.attr.items()] + \
                [f"{i}:D:{Dtypes_names[j]}" for i,j in self.datasets.items()]
            out.write('\t'.join(cols) + '\n')

    def load_header(self):
        """
        Load header, attributes and datasets information for SZ file
        """
        attributes = []
        datasets = []
        index = []
        with open(self.idx_path, 'r') as f:
            # Get first row of header
            header = json.loads(f.readline().rstrip().lstrip('#'))

            # Get attributes and datasets
            cols = f.readline().rstrip().split('\t')
            for col in cols[2:]:
                key_id, key_group, key_dtype = col.split(':')
                if key_group == 'A':
                    attributes.append((key_id, Dtypes[key_dtype]))
                elif key_group == "D":
                    datasets.append((key_id, Dtypes[key_dtype]))
                else:
                    raise KeyError(f"Unsupported data type: {key_group}")

            # Get index dataframe
            for line in f:
                index.append(line.rstrip().split('\t'))

        self.header = header
        self.attr = OrderedDict(attributes)
        self.datasets = OrderedDict(datasets)

        # Convert into pandas dataframe and parser dtype
        self.idx = pd.DataFrame(index)
        self.idx.columns = ['LENGTH', 'OFFSET'] + list(self.attr) + list(self.datasets)
        self.idx['LENGTH'] = self.idx['LENGTH'].astype(int)
        self.idx['OFFSET'] = self.idx['OFFSET'].astype(int)

    def put(self, *data):
        """
        Put new data into queue for compressing and saving

        Args:
            *data: data must be in specific order consistent to the attributes and datasets.
                The default SZ contains the following attributes:

                          ID: read ID, str
                      Offset: Offset for raw current data, np.int32
                    Raw_unit: Raw unit for raw current data, np.float32

                And the following datasets:

                        Raw: Raw current data points, np.uint32
                      Fastq: Basecalled fastq, str
                       Move: move tables, np.uint16
        """
        self.q_in.put(data)

    def encode(self, data):
        """
        Encode input data into SVB-ZSTD compressed binary format
        Args:
            data: data acquired from queue

        Returns:
            encoded, byte str,
                byte str of compressed datasets
            idx, list,
                list of items to be stored in index
        """
        record = self.record(*data)
        dat_encoded = []
        idx_encoded = []

        # Store indices
        for attr_id, attr_dtype in self.attr.items():
            idx_encoded.append(str(attr_dtype(getattr(record, attr_id))))

        # Compress datasets according to the dtypes
        for dataset_id, dataset_dtype in self.datasets.items():
            if dataset_dtype in Svb_dtypes:
                d_bstr, d_size, _ = svb_encode(np.array(getattr(record, dataset_id)).astype(dataset_dtype))
            elif dataset_dtype == str:
                d_bstr = str_encode(dataset_dtype(getattr(record, dataset_id)))
                d_size = ''
            else:
                d_bstr = np.array(getattr(record, dataset_id)).astype(dataset_dtype).tobytes()
                d_size = ''

            d_encoded = zstd_encode(d_bstr)
            dat_encoded.append(d_encoded)
            idx_encoded.append(f"{len(d_encoded)}:{d_size}")

        # Convert encoded into simple binary format
        encoded = b''.join(dat_encoded)

        return encoded, idx_encoded

    def encoder(self):
        """
        Process for compressing input data and pass to SZ writer
        """
        while True:
            raw = self.q_in.get()
            if raw is None:
                break
            dat_line, idx_line = self.encode(raw)
            self.q_out.put((dat_line, idx_line))
        self.q_out.put(None)

    def writer(self):
        """
        Process for handling compressed data and save it to disk
        """
        n_alive = self.threads
        cursor = 0
        with open(self.idx_path, 'a') as idx_out, open(self.dat_path, 'wb') as dat_out:
            while True:
                # Wait for all encode workers to stop
                if n_alive == 0:
                    break
                res = self.q_out.get()
                if res is None:
                    n_alive -= 1
                    continue

                # Record cursor position everytime
                dat_line, idx_line = res
                dat_out.write(dat_line)
                shift, length = cursor, len(dat_line)
                cursor += length
                idx_line = [str(length), str(shift), ] + idx_line
                idx_out.write('\t'.join(idx_line) + '\n')

    def get(self, idx):
        """
        Get specified index from compressed SZ data

        Args:
            idx: int / list
                numeric int or list of integer index, items will be acquired using 'df.iloc';

        Returns:
            list of Record (namedTuple), containing attributes and datasets information specified in the SZ header

        """
        reads = []
        rows = self.idx.iloc[[idx]] if isinstance(idx, int) else self.idx.iloc[idx]
        rows = rows.sort_values(by='OFFSET')

        zdc = ZstdDecompressor() if self.decompressor is None else self.decompressor
        f = open(self.dat_path, 'rb') if self.fh is None else self.fh

        for _, row in rows.iterrows():
            offset, length = row['OFFSET'], row['LENGTH']
            attr = row[2:2+len(self.attr)].tolist()

            f.seek(offset)
            encoded = f.read(length)

            datasets = []
            p = 0
            for d_name, d_col in zip(self.datasets, row[2+len(self.attr):]):
                d_size, d_shape = d_col.split(':')
                bstr = zstd_decode(encoded[p:p+int(d_size)], zdc)
                p += int(d_size)

                d_dtype = self.datasets[d_name]
                if d_dtype in Svb_dtypes:
                    d_decoded = svb_decode(bstr, int(d_shape), dtype=d_dtype)
                elif d_dtype == str:
                    d_decoded = str_decode(bstr)
                else:
                    d_decoded = np.frombuffer(bstr, dtype=d_dtype)
                datasets.append(d_decoded)
            record = attr + datasets
            reads.append(self.record(*record))
        return reads

    def close(self):
        """
        Functions for closing and wanting for process to end.
        """
        if self.mode == 'w':
            for _ in np.arange(self.threads):
                self.q_in.put(None)
            for p in self.pool:
                p.join()
            self.worker.join()

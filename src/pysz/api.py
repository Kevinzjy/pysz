import json
import numpy as np
from pathlib import Path
from pysz import __version__
from pysz.compression import svb_decode, svb_encode, zstd_decode, zstd_encode, str_decode, str_encode
from pysz.utils import mkdir, assert_file_exists, assert_dir_exists
from collections import OrderedDict, namedtuple
from multiprocessing import Manager, Process, Pool


Attributes = [
    ('ID', str),
    ('Offset', np.int32),
    ('Raw_unit', np.uint32),
]
Datasets = [
    ('Raw', np.uint32),
    ('Fastq', np.uint32),
    ('Move', np.uint32),
    ('Norm', np.uint32),
]

Dtypes_names = {
    str: "S",
    np.int16: "I16", np.int32: "I32",
    np.uint16: "U16", np.uint32: "U32",
    np.float16: "F16", np.float32: "F32",
}
Dtypes = {j: i for i, j in Dtypes_names.items()}

Svb_dtypes = [np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64]


class CompressedFile(object):

    def __init__(self, data_dir, mode:str="r", overwrite:bool=False):
        self.dir = data_dir if isinstance(data_dir, Path) else Path(data_dir)
        self.idx_path = self.dir / "index"
        self.dat_path = self.dir / "dat"
        self.mode = mode
        if self.mode not in ['r', 'w']:
            raise KeyError("Mode should be either 'w' or 'r'.")
        self.overwrite = overwrite

        if mode == "r":
            assert_dir_exists(data_dir)
            assert_file_exists(self.idx_path)
            assert_file_exists(self.dat_path)
            self.idx_fh, self.dat_fh = None, None
        else:
            mkdir(data_dir, overwrite=overwrite)
            self.idx_fh = open(self.idx_path, 'w')
            self.dat_fh = open(self.dat_path, 'wb')

        self.header = {"version": __version__}
        self.attr = OrderedDict(Attributes)
        self.datasets = OrderedDict(Datasets)
        self.record = namedtuple("Record", list(self.attr) + list(self.datasets))

    def write_template(self, header:list=None, attributes:list=None, datasets:list=None):
        if self.mode != 'w':
            raise RuntimeError("")
        if header is not None:
            self.header = dict(header)
        if attributes is not None:
            self.attr = OrderedDict(attributes)
        if datasets is not None:
            self.datasets = OrderedDict(datasets)
        self.record = namedtuple("Record", list(self.attr) + list(self.datasets))

        self.write_header()
        self.write_columns()

    def write_header(self):
        header_str = json.dumps(self.header)
        self.idx_fh.write("#" + header_str + '\n')

    def write_columns(self):
        cols = ["NAME", "LENGTH", "OFFSET",] + \
               [f"{i}:A:{Dtypes_names[j]}" for i,j in self.attr.items()] + \
               [f"{i}:D:{Dtypes_names[j]}" for i,j in self.datasets.items()]
        self.idx_fh.write('\t'.join(cols) + '\n')

    def write_record(self, *data):
        record = self.record(*data)
        dat_encoded = []
        idx_encoded = []

        for attr_id, attr_dtype in self.attr.items():
            idx_encoded.append(attr_dtype(getattr(record, attr_id)))

        for dataset_id, dataset_dtype in self.datasets.items():
            if dataset_dtype in Svb_dtypes:
                d_bstr, d_size, _ = svb_encode(np.array(getattr(record, dataset_id)).astype(dataset_dtype))
                d_encoded = zstd_encode(d_bstr)
                # d_encoded = d_bstr
                idx_encoded.append(f"{len(d_encoded)}:{d_size}")
            elif dataset_dtype != str:
                d_bstr = np.array(getattr(record, dataset_id)).astype(dataset_dtype).tobytes()
                d_encoded = zstd_encode(d_bstr)
                # d_encoded = d_bstr
                idx_encoded.append(f"{len(d_encoded)}:")
            else:
                d_bstr = str_encode(dataset_dtype(getattr(record, dataset_id)))
                # d_encoded = zstd_encode(d_bstr)
                d_encoded = d_bstr
                idx_encoded.append(f"{len(d_encoded)}:")
            dat_encoded.append(d_encoded)
        encoded = b''.join(dat_encoded)
        return encoded, idx_encoded

    def close(self):
        self.idx_fh.close()
        self.dat_fh.close()




# import sys
# import copy
# import json

# import torch
# import numpy as np
# import pandas as pd
# from zstandard import ZstdDecompressor
# from torch.utils.data import Dataset, DataLoader, random_split
#
# from Hetero_seq.compression import zstd_encode, zstd_decode
# from Hetero_seq.compression import svb_encode, svb_decode
# from Hetero_seq.compression import str_encode, str_decode
# from Hetero_seq.utils import ensure_file_exists, scale_events, get_mad
#
#
# class ChunkRecord(object):
#     def __init__(self, offset, raw_unit, raw_data, fastq, move, raw_smoothed=None):
#         self.raw_data = raw_data
#         self.offset = offset
#         self.raw_unit = raw_unit
#         self.fastq = fastq
#         self.move = move
#         self.smoothed = raw_smoothed
#
#     def encode(self, cctx, alphabet, max_len):
#         raw_bstr, raw_size, raw_dtype = svb_encode(self.raw_data.astype(np.uint32))
#         raw_encoded = zstd_encode(raw_bstr, cctx)
#
#         fastq_bstr, fastq_size, fastq_dtype = svb_encode(np.array([alphabet[x] for x in f"{self.fastq:<{max_len}}"], dtype=np.uint16))
#         fastq_encoded = zstd_encode(fastq_bstr, cctx)
#
#         move_bstr, move_size, move_dtype = svb_encode(self.move.astype(np.uint16))
#         move_encoded = zstd_encode(move_bstr, cctx)
#
#         if self.smoothed is not None:
#             smooth_bstr, smooth_size, smooth_dtype = svb_encode(self.smoothed.astype(np.uint32))
#             smooth_encoded = zstd_encode(smooth_bstr, cctx)
#         else:
#             smooth_encoded = b''
#             smooth_size = 0
#
#         content = [
#             str_encode(str(self.offset)), str_encode(str(self.raw_unit)),
#             raw_encoded, fastq_encoded, move_encoded, smooth_encoded
#         ]
#         encoded = b''.join(content)
#         col_size = [len(i) for i in content]
#         array_size = [0, 0, raw_size, fastq_size, move_size, smooth_size]
#         return encoded, col_size, array_size
#
#
# class ChunkData(object):
#     def __init__(self, zdc, encoded, col_size, array_size):
#         i = 0
#         content = []
#         for x in col_size:
#             content.append(encoded[i:i+x])
#             i += x
#
#         self.offset = float(str_decode(content[0]))
#         self.raw_unit = float(str_decode(content[1]))
#
#         raw_bstr = zstd_decode(content[2], zdc)
#         self.raw_data = svb_decode(raw_bstr, array_size[2], dtype=np.uint32)
#
#         fastq_bstr = zstd_decode(content[3], zdc)
#         self.fastq = svb_decode(fastq_bstr, array_size[3], dtype=np.uint16)
#
#         move_bstr = zstd_decode(content[4], zdc)
#         self.move = svb_decode(move_bstr, array_size[4], dtype=np.uint16)
#
#         self.current = np.array((self.raw_data + self.offset) * self.raw_unit, dtype=np.float32)
#
#         if array_size[5] == 0:
#             self.smoothed = None
#             self.current_smoothed = None
#         else:
#             smooth_bstr = zstd_decode(content[5], zdc)
#             self.smoothed = svb_decode(smooth_bstr, array_size[5], dtype=np.uint32)
#             self.current_smoothed = np.array((self.smoothed + self.offset) * self.raw_unit, dtype=np.float32)
#
#
# class ChunkFile(object):
#     def __init__(self, file_name, multiprocessing=False):
#         self.fn = ensure_file_exists(file_name)
#
#         self.fh = open(self.fn, 'rb')
#         self.multiprocessing = multiprocessing
#         self.decompressor = None if self.multiprocessing else ZstdDecompressor()
#
#         self.idx_fn = self.fn.with_suffix(".chunk5.idx")
#         self.idx = pd.read_csv(self.idx_fn, sep="\t", comment="#")
#         with open(self.idx_fn, 'r') as f:
#             self.header = json.loads(f.readline().lstrip("#"))
#
#     def close(self):
#         self.fh.close()
#         return 0
#
#     def __getitem__(self, idx):
#         reads = []
#         rows = self.idx.iloc[[idx]] if isinstance(idx, int) else self.idx.iloc[idx]
#         rows = rows.sort_values(by='Offset')
#
#         if self.multiprocessing:
#             f = open(self.fn, 'rb')
#             zdc = ZstdDecompressor()
#         else:
#             f = self.fh
#             zdc = self.decompressor
#
#         for _, row in rows.iterrows():
#             offset, length = int(row['Offset']), int(row['Length'])
#             col_size = [int(i) for i in row['Col_size'].split(',')]
#             array_size = [int(i) for i in row['Array_size'].split(',')]
#             f.seek(offset)
#             encoded = f.read(length)
#             read = ChunkData(zdc, encoded, col_size, array_size)
#             reads.append(read)
#
#         if self.multiprocessing:
#             f.close()
#
#         return reads
#
#
# class ChunkDataset(Dataset):
#
#     def __init__(self, chunk_file, multiprocessing=True):
#         self.chunk_file = ChunkFile(chunk_file, multiprocessing)
#
#     def __len__(self):
#         return self.chunk_file.idx.shape[0]
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         raw_data, fastq, move, smoothed = [], [], [], []
#         reads = copy.deepcopy(self.chunk_file[idx])
#         for read in reads:
#             _median, _mad = get_mad(read.current)
#             raw_data.append(scale_events(read.current, median=_median, mad=_mad))
#             fastq.append(read.fastq)
#
#             # One-hot encoding
#             _move = torch.LongTensor(read.move.reshape(-1, 1).astype(np.int16))
#             _move = torch.zeros(read.move.shape[0], 2).scatter_(1, _move, 1)
#             move.append(_move)
#
#             smoothed.append(scale_events(read.current_smoothed, median=_median, mad=_mad))
#
#         raw_data = np.concatenate([raw_data], dtype=np.float32)
#         raw_data = torch.from_numpy(raw_data.reshape(-1, raw_data.shape[1], 1))
#
#         fastq = torch.from_numpy(np.concatenate([fastq], dtype=np.int64))
#         move = torch.stack(move)
#
#         smoothed = np.concatenate([smoothed], dtype=np.float32)
#         smoothed = torch.from_numpy(smoothed.reshape(-1, smoothed.shape[1], 1))
#
#         # label_lengths = (references > 0).sum(axis=1)
#         sample = {
#             'raw_data': raw_data,
#             'fastq': fastq,
#             'move': move,
#             'smoothed': smoothed,
#         }
#         return sample
#
#
# def load_chunk_dataset(chunk_file, batch_size, num_workers, prefetch_factor):
#     signal_dataset = ChunkDataset(chunk_file, multiprocessing=True)
#     signal_loader = DataLoader(
#         signal_dataset, batch_size=batch_size, num_workers=num_workers,
#         persistent_workers=True, shuffle=True, pin_memory=True, prefetch_factor=prefetch_factor,
#     )
#     return signal_dataset, signal_loader

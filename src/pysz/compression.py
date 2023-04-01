import numpy as np
import streamvbyte
from zstandard import ZstdDecompressor, ZstdCompressor


def str_to_bytes(bytes_or_str):
    """
    Convert str to byte string
    :param bytes_or_str:
    :return:
    """
    if isinstance(bytes_or_str, str):
        value = bytes_or_str.encode('utf-8')
    else:
        value = bytes_or_str
    return value


def bytes_to_str(bytes_or_str):
    """
    Convert byte string to string
    :param bytes_or_str:
    :return:
    """
    if isinstance(bytes_or_str, bytes):
        value = bytes_or_str.decode('utf-8')
    else:
        value = bytes_or_str
    return value


def str_encode(data):
    """
    Convert str to byte
    :param data:
    :return:
    """
    return str_to_bytes(data)


def str_decode(data):
    """
    Convert byte to str
    :param data:
    :return:
    """
    return bytes_to_str(data)


def is_gzipped(file_name):
    """
    Examine whether a file is gzipped.
    Reference: https://stackoverflow.com/questions/3703276/how-to-tell-if-a-file-is-gzip-compressed
    """
    with open(file_name, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'


def zstd_encode(bstr, cctx=None):
    """
    Compress byte string using Zstd library
    :param data: byte str
        data for compression
    :param cctx: zstandard.ZstdCompressor
        instance of zstd compressor
    :return: byte str
        compressed data
    """
    _cctx = ZstdCompressor() if cctx is None else cctx
    compressed = _cctx.compress(bstr)
    return compressed


def zstd_decode(compressed, zdc=None):
    """
    Decompress zstd compressed data
    :param compressed:
    :param zdc:
    :return:
    """
    _zdc = ZstdDecompressor() if zdc is None else zdc
    decompressed = _zdc.decompress(compressed)
    return decompressed


def svb_encode(data):
    """
    Compress data using streamvbyte algorithm, int16, uint16, int32, uint32 is supported
    :param data:
    :return:
    """
    if data.dtype not in [np.uint16, np.int16, np.int32, np.uint32]:
        raise TypeError("Streamvbyte only support int16/uint16/int32/uint32 formats!")
    compressed = streamvbyte.encode(data).tobytes()
    return compressed, data.shape[0], data.dtype


def svb_decode(data, size, dtype=np.int32):
    """
    Decompress streamvbyte encoded data
    :param data:
    :param size:
    :param dtype:
    :return:
    """
    decoded = np.frombuffer(data, dtype=np.uint8)
    decompressed = streamvbyte.decode(decoded, size, dtype=dtype)
    return decompressed
import sys
import warnings
from pathlib import Path

def mkdir(dir_name, overwrite=False):
    dir_path = dir_name if isinstance(dir_name, Path) else Path(dir_name)
    if dir_path.exists():
        if dir_path.is_dir():
            if overwrite is True:
                sys.stderr.write(f'Overwriting existed {dir_name}\n')
            else:
                raise OSError(f'{dir_name} already exists, please use --overwrite to force overwrite')
        else:
            raise OSError(f'{dir_name} conflict with existing files!')
    else:
        dir_path.mkdir()
    return dir_path


def assert_dir_exists(dir_name):
    dir_path = dir_name if isinstance(dir_name, Path) else Path(dir_name)
    if dir_path.exists() and dir_path.is_dir():
        return dir_path
    raise OSError(f"Directory not found: {dir_path}.")


def assert_file_exists(file_name):
    file_path = file_name if isinstance(file_name, Path) else Path(file_name)
    if file_path.exists() and file_path.is_file():
        return file_path
    raise OSError(f"File not found: {file_path}.")


def error_callback(e):
    sys.stderr.write(f"Error in worker process: {e.__cause__}: {e}")
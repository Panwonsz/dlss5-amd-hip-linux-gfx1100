"""Exercise the real preload constructor in a fresh process, with no HIP calls."""
import ctypes
import os
from pathlib import Path
import sys

library = ctypes.CDLL(sys.argv[1])
libc = ctypes.CDLL(None)
libc.getenv.argtypes = [ctypes.c_char_p]
libc.getenv.restype = ctypes.c_char_p
value = libc.getenv(b'DLSS5_HIP_BRIDGE')
assert value and int(value, 16), 'missing process-local bridge'
address = int(value, 16)
assert ctypes.c_uint64.from_address(address).value == 0x3150494853534C44
# strace is deliberately unnecessary: this catches the old writer in the source,
# and the fresh process exercises the constructor which used to publish the file.
source = Path(__file__).resolve().parents[1] / 'src/capi.hip'
assert 'DLSS5_HIP_ADDR_FILE' not in source.read_text(), 'shared address-file writer remains'
library.dlss5_last_error.restype = ctypes.c_char_p
assert library.dlss5_run(None, None, 0) == -1
assert b'not initialized' in library.dlss5_last_error()
library.dlss5_shutdown()
print('process-local bridge publication and uninitialized API: PASS')

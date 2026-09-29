#!/usr/bin/env python3
"""Host-side twin of OptiScaler's in-process probe (DLSS5_NR_INPROC_PROBE=1), 2026-09-29.

Runs the model on the same synthetic 1920x1080 frame the game-side probe uses and prints the same
output hash, so the two can be compared line for line.

    python3 tests/probe_inproc_host.py --weights <dir>                       # libdlss5_hip.so directly
    python3 tests/probe_inproc_host.py --weights <dir> --shim libdlss5_hip_preload.so
                                         # through the preload shim's bridge, as dlss5_hip.dll calls it
"""
import argparse, ctypes, os, sys, time
from pathlib import Path
import numpy as np

W, H = 1920, 1080


def frame():
    y, x, c = np.meshgrid(np.arange(H), np.arange(W), np.arange(4), indexing='ij')
    return (((x * 7 + y * 13 + c * 29) % 256).astype(np.float32) * np.float32(0.00390625)).ravel()


def digest(out):
    # sum over i of bits[i] * (2i + 1), mod 2^64 -- the game-side probe computes the same
    bits = out.view(np.uint32).astype(np.uint64)
    weights = np.arange(bits.size, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    with np.errstate(over='ignore'):
        return int(np.sum(bits * weights, dtype=np.uint64))


class Bridge(ctypes.Structure):
    _fields_ = [('magic', ctypes.c_uint64),
                ('init', ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int)),
                ('run', ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)),
                ('shutdown', ctypes.CFUNCTYPE(None)),
                ('last_error', ctypes.CFUNCTYPE(ctypes.c_char_p))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--library', default=str(Path(__file__).resolve().parents[1] / 'libdlss5_hip.so'))
    ap.add_argument('--shim')
    a = ap.parse_args()

    if a.shim:
        os.environ.setdefault('DLSS5_HIP_LIBRARY', str(Path(a.library).resolve()))
        ctypes.CDLL(str(Path(a.shim).resolve()))
        libc = ctypes.CDLL(None)
        libc.getenv.restype = ctypes.c_char_p
        libc.setenv(b'DLSS5_HIP_LIBRARY', os.environ['DLSS5_HIP_LIBRARY'].encode(), 1)
        address = int(libc.getenv(b'DLSS5_HIP_BRIDGE'), 16)
        b = Bridge.from_address(address)
        assert b.magic == 0x3150494853534C44, 'bridge magic'
        init, run, shutdown, last_error = b.init, b.run, b.shutdown, b.last_error
        print('via preload shim bridge at', hex(address))
    else:
        lib = ctypes.CDLL(str(Path(a.library).resolve()))
        lib.dlss5_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.dlss5_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
        lib.dlss5_last_error.restype = ctypes.c_char_p
        init, run, shutdown, last_error = lib.dlss5_init, lib.dlss5_run, lib.dlss5_shutdown, lib.dlss5_last_error

    t = time.perf_counter()
    if init(a.weights.encode(), 0) != 0:
        sys.exit('dlss5_init failed: %s' % last_error().decode(errors='replace'))
    print('init ok in %.0f ms' % ((time.perf_counter() - t) * 1e3))
    inp = frame()
    out = np.zeros(W * H * 3, np.float32)
    times, digests = [], []
    for _ in range(5):
        t = time.perf_counter()
        if run(inp.ctypes.data, out.ctypes.data, 0) != 0:
            sys.exit('dlss5_run failed: %s' % last_error().decode(errors='replace'))
        times.append((time.perf_counter() - t) * 1e3)
        digests.append(digest(out))
    shutdown()
    print('5 runs %s ms (wall, incl. host copies) | input digest %016x | output digest %016x | replay %s'
          % (' '.join('%.1f' % v for v in times), digest(inp), digests[0],
             'equal' if len(set(digests)) == 1 else 'DIFFERS'))


if __name__ == '__main__':
    main()

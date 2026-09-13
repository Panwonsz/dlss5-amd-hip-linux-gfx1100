#!/usr/bin/env python3
"""Run the local 71-block HIP port on a PNG/JPEG/PPM, not an upscaler fallback."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def runtime_paths(root):
    """Resolve either the source checkout or the extracted local archive."""
    root = Path(root)
    for package, library in ((root / 'linux', root / 'hip/libdlss5_hip.so'),
                             (root, root / 'bin/libdlss5_hip.so')):
        if (package / 'dlssnr').is_dir() and library.is_file():
            return package, library
    raise RuntimeError('HIP runtime not found; build it with make -C hip game or extract the full archive')


def save_image(path, values):
    import numpy as np
    from PIL import Image
    path = Path(path)
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[2] != 3 or not np.isfinite(values).all():
        raise ValueError('RGB output must be finite HxWx3')
    fmt = {'.png': 'PNG', '.jpg': 'JPEG', '.jpeg': 'JPEG', '.ppm': 'PPM'}.get(path.suffix.lower())
    if fmt is None:
        raise ValueError('Output extension must be .png, .jpg or .ppm')
    pixels = np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8)
    with path.open('xb') as stream:
        if fmt == 'PPM':
            h, w = pixels.shape[:2]
            stream.write(f'P6\n{w} {h}\n255\n'.encode() + pixels.tobytes())
        else:
            Image.fromarray(pixels).save(stream, format=fmt)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--weights', type=Path, help='Complete coefficient folder')
    source.add_argument('--nvidia-dll', type=Path, help='Locally owned nvngx_dlssnr.dll 310.8.0.0')
    p.add_argument('--allow-derived-layouts', action='store_true',
                   help='Explicitly accept reconstructed AMD-consumer layouts; not NVIDIA equivalence')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--runs', type=int, default=1)
    p.add_argument('--trace', action='store_true')
    args = p.parse_args(argv)
    if args.output.exists():
        p.error('Output already exists; choose a new path')
    if args.device < 0 or not 0 <= args.seed <= 0xffffffff or not 1 <= args.runs <= 1000:
        p.error('Invalid device, uint32 seed or run count')
    import numpy as np
    from PIL import Image
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent if script_dir.name == 'hip' else script_dir
    package_root, library_path = runtime_paths(root)
    sys.path.insert(0, str(package_root))
    from dlssnr.convert_dll import convert_nvidia_dll, validate_cache
    if args.nvidia_dll:
        mode = 'amd-consumer-derived' if args.allow_derived_layouts else None
        kwargs = {'layout_mode': mode} if mode else {}
        weights = convert_nvidia_dll(args.nvidia_dll, **kwargs)
    else:
        weights = args.weights.expanduser().resolve()
    manifest = None
    if (weights / 'manifest.json').is_file():
        manifest = json.loads((weights / 'manifest.json').read_text())
        mode = manifest.get('layout_mode')
        if mode == 'amd-consumer-derived' and not args.allow_derived_layouts:
            p.error('Reconstructed cache requires --allow-derived-layouts')
        validate_cache(weights, **({'layout_mode': mode} if mode else {}))
    image = Image.open(args.input).convert('RGBA')
    if image.size != (1920, 1080):
        p.error('This port requires a 1920x1080 image; no implicit rescaling')
    rgba = np.ascontiguousarray(np.asarray(image, dtype=np.float32) / 255.0)
    rgb = np.empty((1080, 1920, 3), dtype=np.float32)
    if args.trace:
        os.environ['DLSS5_HIP_TRACE'] = '1'
    lib = ctypes.CDLL(str(library_path))
    fp = ctypes.POINTER(ctypes.c_float)
    lib.dlss5_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.dlss5_run.argtypes = [fp, fp, ctypes.c_uint32]
    lib.dlss5_last_error.restype = ctypes.c_char_p
    start = time.perf_counter()
    if lib.dlss5_init(os.fsencode(weights), args.device):
        raise RuntimeError(lib.dlss5_last_error().decode())
    init_ms = (time.perf_counter() - start) * 1000
    timings, previous = [], None
    try:
        for _ in range(args.runs):
            start = time.perf_counter()
            if lib.dlss5_run(rgba.ctypes.data_as(fp), rgb.ctypes.data_as(fp), args.seed):
                raise RuntimeError(lib.dlss5_last_error().decode())
            timings.append((time.perf_counter() - start) * 1000)
            if not np.isfinite(rgb).all():
                raise RuntimeError('Nonfinite inference output')
            if previous is not None and not np.array_equal(previous, rgb):
                raise RuntimeError('Identical input/seed replay failed')
            previous = rgb.copy()
        save_image(args.output, rgb)
    finally:
        lib.dlss5_shutdown()
    report = {'input': str(args.input.resolve()), 'output': str(args.output.resolve()),
              'weights': str(weights), 'layout_mode': manifest.get('layout_mode') if manifest else 'external-tables',
              'nvidia_equivalence_verified': False, 'blocks': 71, 'seed': args.seed,
              'model_init_ms': init_ms, 'run_wall_ms': timings, 'replay_equal': True,
              'finite': True, 'mean_abs_change': float(np.abs(rgb - rgba[:, :, :3]).mean()),
              'library_sha256': hashlib.sha256(library_path.read_bytes()).hexdigest(),
              'output_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'note': 'Offline display-encoded image inference; not a game/HDR or NVIDIA reference comparison.'}
    report_path = args.output.with_suffix(args.output.suffix + '.json')
    with report_path.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'infer_image: {exc}', file=sys.stderr)
        raise SystemExit(1)

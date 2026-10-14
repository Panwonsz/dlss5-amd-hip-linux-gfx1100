#!/usr/bin/env python3
"""DLSS5_F16_ACT=1 against the bit-exact model, on real images (2026-10-14).

Runs hip-network70 twice per image -- exact, then with the C32 blocks on f16 activations -- and reports
how far apart the two outputs are, plus side-by-side PNGs for looking at.

    python3 hip/tests/compare_f16act.py --weights "$W" shot1.png shot2.png ...

Images: game screenshots at 1920x1080 work best (others are resized). Per image it prints
  max |d|, mean |d|  over the RGB outputs in [0,1]
  PSNR               of the f16-activation output against the exact one (higher = closer; >45 dB is
                     usually invisible, >50 dB very hard to see)
  8-bit changed      share of output values that differ after rounding to 8 bits (what a monitor shows)
and writes <out>/<name>_exact.png, _f16act.png, _diff16.png (|d| x 16, so faint differences show).
"""
import argparse, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

W, H = 1920, 1080


def to_rgba32f(png, raw):
    img = Image.open(png).convert('RGB')
    if img.size != (W, H):
        img = img.resize((W, H), Image.LANCZOS)
    rgb = np.asarray(img, dtype=np.float32) / np.float32(255)
    rgba = np.concatenate([rgb, np.ones((H, W, 1), np.float32)], axis=2)
    rgba.astype('<f4').tofile(raw)


def run(bench, weights, raw_in, raw_out, f16act):
    env = dict(os.environ, DLSS5_F16_ACT='1' if f16act else '0')
    r = subprocess.run([bench, '--weights', weights, '--input', raw_in, '--output', raw_out,
                        '--runs', '3', '--seed', '0'], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f'{bench} failed ({"f16act" if f16act else "exact"}):\n{r.stdout}\n{r.stderr}')
    ms = [float(t.split('=')[1]) for line in r.stdout.splitlines() for t in line.split() if t.startswith('gpu_ms=')]
    return np.fromfile(raw_out, '<f4').reshape(H, W, 3), min(ms) if ms else float('nan')


def save(path, rgb):
    Image.fromarray(np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--bench', default=str(Path(__file__).resolve().parents[1] / 'hip-network70'))
    ap.add_argument('--out', default='f16act-compare')
    ap.add_argument('images', nargs='+')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f'{"image":28} {"max|d|":>9} {"mean|d|":>10} {"PSNR dB":>8} {"8-bit changed":>14} {"exact ms":>9} {"f16 ms":>8}')
    with tempfile.TemporaryDirectory() as tmp:
        for i, png in enumerate(a.images):
            raw = f'{tmp}/in{i}.raw'
            to_rgba32f(png, raw)
            exact, ms_e = run(a.bench, a.weights, raw, f'{tmp}/e{i}.raw', False)
            f16, ms_f = run(a.bench, a.weights, raw, f'{tmp}/f{i}.raw', True)
            d = np.abs(exact.astype(np.float64) - f16)
            mse = float(np.mean(d * d))
            psnr = float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)
            q = np.clip(np.rint(exact * 255), 0, 255) != np.clip(np.rint(f16 * 255), 0, 255)
            name = Path(png).stem
            print(f'{name[:28]:28} {d.max():9.5f} {d.mean():10.6f} {psnr:8.2f} {100 * q.mean():13.3f}% '
                  f'{ms_e:9.2f} {ms_f:8.2f}')
            save(out / f'{name}_exact.png', exact)
            save(out / f'{name}_f16act.png', f16)
            save(out / f'{name}_diff16.png', np.clip(d * 16, 0, 1))
    print(f'images in {out}/')


if __name__ == '__main__':
    main()

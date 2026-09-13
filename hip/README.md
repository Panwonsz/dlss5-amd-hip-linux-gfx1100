# DLSS5-AMD — native HIP backend

Port of [lmxxf/dlss5-on-amd-9070xt-porting](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting),
using HIP and rocWMMA on RDNA 4 (`gfx1201`), without D3D12 wave-matrix support.
The source contracts are pinned to upstream `15799b1600d57b849597a44be53ac892b7a2faea`.

**The complete 71-block network runs offline:** preblock, Swin encoder, eight
640×1024 ViT blocks, decoder with skip connections, postblock and RGB head.
Input is 1920×1080; internal reflection padding extends to 1920×1152.
All coefficients come from the user's NVIDIA DLL. No proprietary AMD inference
DLL or fatbin is loaded.

## Build and verify

From the repository root, with ROCm/HIP 7, rocWMMA headers and an x64 MinGW
compiler available for the small PE bridge:

```sh
make -C hip -j3 game hip-network70
```

The public snapshot excludes the local regression suites. See `docs/BUILD.md`
for build details and executable verification commands. Passing offline inference
does not establish gameplay or NVIDIA numerical/visual equivalence.

## Run an image

Python needs NumPy and Pillow. The input must be 1920×1080; this command never
silently rescales it or overwrites an existing output:

```sh
python3 hip/infer_image.py input.png output.png \
  --nvidia-dll /absolute/path/to/nvngx_dlssnr.dll \
  --allow-derived-layouts --runs 2
```

Alternatively use `--weights /absolute/path/to/complete/tables`. The output has a
JSON sidecar with actual timing, hashes, finiteness and same-seed replay results.
The low-level `hip/hip-network70 --help` documents packed float input/output and
per-block tracing.

Only the SHA-pinned NVIDIA DLL 310.8.0.0 is accepted by the converter. Its complete
cache contains 223 tables. Some C32/attention layouts and the ViT bridge are
**AMD-consumer-derived**, not comparisons with the missing original NVIDIA maps.
`--allow-derived-layouts` explicitly accepts this experimental mode. Strict
conversion refuses incomplete maps instead of supplying fallback matrices.

## Host API and Proton boundary

`include/dlss5_capi.h` defines the synchronous host ABI:

- `dlss5_run`: packed 1080p display-encoded RGBA to RGB, no history.
- `dlss5_run_frame`: explicit linear/display-sRGB codec, paper white, transfer,
  color strength, reset and optional full XY motion rectangle with caller-supplied
  UV scale. History reprojection uses the published fast five-tap filter.
- `dlss5_hip.dll`: Win64-to-SysV forwarding, using only Wine's live process-local
  environment. No shared pointer files; the native library stays resident.

The live ReShade path records a private marker at the FSR boundary. The matching
modified vkd3d runtime submits the producer prefix, waits on its submission worker,
runs HIP using CPU staging, then submits the consumer suffix. It resets history
because motion readback is not wired there. F6 toggles bypass. The game still waits
for inference: **this is a slow proof of concept needing substantial optimization**.
General gameplay stability, HDR and NVIDIA equivalence remain unverified.
No frame generation is implemented. See the root README for installation.

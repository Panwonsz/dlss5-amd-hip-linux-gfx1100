# Building the HIP proof of concept

This is an unoptimized experimental port, not a production gameplay release.
The public source snapshot contains the runtime and installer; local test suites,
game files, captures, caches and NVIDIA weights are excluded.

## HIP and ReShade add-on

Prerequisites: ROCm/HIP 7 with rocWMMA headers in `/opt/rocm`, Python 3.10+,
a recent x86_64 llvm-mingw toolchain on PATH, Git, and build dependencies.

```sh
make -C hip -j3 game hip-network70 MINGW_CC=x86_64-w64-mingw32-gcc
linux/install.sh build-addon
```

The add-on helper downloads MinHook/ReShade headers when absent. The binary
release needs neither a compiler nor a D3D12 SM 6.10 runtime.

For the exact vendored headers used in this release, extract the accompanying
`addon-build-deps.tar.gz` into the source root. Its `third_party/` folder includes
licenses. Add `DLSS5_EXTRA_INCLUDE="$PWD/third_party/mingw-headers"` when building
the add-on with those headers.

## Modified vkd3d-proton

The release supplies `vkd3d-proton-poc-source.tar.gz`: complete source and bundled
subprojects corresponding to the shipped DLL pair, including LGPL notices,
a record of modified files and a portable cross file. Follow its `BUILD-POC.md`.
The source was derived from HansKristian-Work/vkd3d-proton
`35bdee1435c94f8c3548725fcb046595b263bd7e`, with local ordering and local callback
extensions. It is **not** an unmodified upstream build. Its embedded version
string was generated inside the parent worktree and is not an upstream revision
identifier; use the supplied source inventory and binary hashes.

For `linux/build_release.py`, place the extracted source at
`linux/build/live/vkd3d-proton`, and its build directory at
`linux/build/live/build-vkd3d`. The packager expects both `libs/d3d12/d3d12.dll`
and `libs/d3d12core/d3d12core.dll`. The latter is the LGPL vkd3d implementation,
not a Microsoft D3D12 runtime.

Obtain ReShade's full-add-on build from https://reshade.me/ (the release used
6.8.0). Place `ReShade64.dll`, its `LICENSE.md` and `PROVENANCE.txt` in
`linux/vendor/reshade/`. The release records the original hashes in its notices.
Then run `python3 linux/build_release.py` to produce `dist/dlss5-amd-hip-linux.tar.gz`.
The separate corresponding-source archives must accompany a published binary release.

## Exercise the actual runtime

```sh
hip/hip-network70 --help
hip/hip-network70 --weights /absolute/path/to/complete/tables --runs 3
```

Without `--input`, the benchmark clearly uses a synthetic gradient image with
your real coefficients. With `--input`, supply 1920×1080 packed float32 RGBA.
The report includes 71-block execution, finiteness, changed pixels and same-seed
replay. These are runtime checks, not an NVIDIA image-reference comparison.

For PNG/JPEG input, install `hip/requirements.txt` in a Python virtual environment
and use the command in `hip/README.md`. Source and binary SHA256 inventories are
published with the release. Do not launch a game merely to test the build.

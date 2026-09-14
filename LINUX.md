# Linux overlay on this branch

GitHub `master` is **unchanged**. It remains the Proton/HIP installer that
wraps [danielblnc/DLSS-NR-on-AMD](https://github.com/danielblnc/DLSS-NR-on-AMD).

This branch (`lmxxf-base`) is a **single squashed snapshot** of
[lmxxf/dlss5-on-amd-9070xt-porting](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting)
initially at `02ad50a`, with local source updates through upstream
`15799b1600d57b849597a44be53ac892b7a2faea`, plus an installer under [`linux/`](linux/)
that reuses game/Proton discovery from `experimental/rx9060-gfx1200`.

`Development/` (≈700 MiB of reverse-engineering notes) is omitted so the
branch stays small. Get it from lmxxf if you need the lab history.

| Path | What |
| --- | --- |
| repo root | lmxxf D3D12 / HLSL / ReShade port |
| [`linux/`](linux/) | Proton installer that deploys that add-on (not danielblnc HIP) |

`linux/` is the Proton installer for this add-on: it copies an lmxxf **user
package** (`dlss5-amd.addon64`, ReShade loader, `DLSS5-AMD/`, Agility SDK) next
to the game `.exe` and writes `.dlssnr-linux/launch.sh` (`WINEDLLOVERRIDES` for
`d3d12` / `dxgi`). It does **not** install danielblnc HIP `setup.exe`.

Shader Model 6.10 `dx::linalg` still needs a driver/runtime that exposes it.
vkd3d-proton often cannot; if the log stays on FSR, that is the current limit.

A native HIP rewrite lives in [`hip/`](hip/) (rocWMMA on `gfx1201`).
The complete 71-block network, codecs and host-motion API run offline.
See [`hip/README.md`](hip/README.md) for builds, tests and image inference.
The complete weight cache explicitly marks reconstructed layouts; it is not a
NVIDIA-equivalence result.

**Proton boundary:** the ReShade readback/writeback path retains synchronous
D3D waits and resets temporal history. It is not validated for live gameplay.
The local installer stages diagnostics; it must not automatically arm a
continuous same-frame wait path. No game installation or launch is part of the
offline verification below.

```sh
make -C hip -j3 all game
make -C hip check
linux/install.sh build-addon
```

To refresh the lmxxf sources later, copy `src/ shaders/ scripts/ tools/` from
lmxxf `main` onto this snapshot. Do not rebase onto lmxxf `main`: that would
reintroduce `Development/` and a 500+ MiB pack.

# Linux overlay

`main` contains the Linux HIP/rocWMMA implementation, the Proton installer under
[`linux/`](linux/), and the supporting D3D12/HLSL reference sources.

| Path | What |
| --- | --- |
| [`hip/`](hip/) | native HIP/rocWMMA 71-block network, offline bench and tests |
| [`linux/`](linux/) | Proton installer that deploys the add-on and launch wrapper |

`linux/` copies the add-on package (`dlss5-amd.addon64`, ReShade loader,
`DLSS5-AMD/`, Agility SDK) next to the game `.exe` and writes
`.dlssnr-linux/launch.sh` (`WINEDLLOVERRIDES` for `d3d12` / `dxgi`). It does
**not** install a proprietary HIP `setup.exe`.

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

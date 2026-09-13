# DLSS5-AMD HIP for Linux — proof of concept

> **This is a slow, experimental proof of concept. It needs substantial optimization and is not ready for normal gameplay.** Expect very low frame rates, high latency and possible rendering problems. It is not an official NVIDIA DLSS implementation or a claim of equivalent image quality.

A Linux HIP/rocWMMA port of [lmxxf's DLSS5-AMD work](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting), with a ReShade add-on, Wine bridge and modified vkd3d-proton submission path. The complete 71-block network runs on AMD `gfx1201`. NVIDIA DLLs and weights are **not included**.

## Download and install

**[Download the prebuilt .tar.gz](https://github.com/guentra/dlss5-amd-hip-linux/releases/download/v0.1.0-poc/dlss5-amd-hip-linux.tar.gz)** · [Release, checksums and corresponding source](https://github.com/guentra/dlss5-amd-hip-linux/releases/tag/v0.1.0-poc)

1. Close the game. Extract the archive inside its directory, keeping the `dlss5-amd-hip-linux` subfolder.
2. Put your legitimately obtained `nvngx_dlssnr.dll` **310.8.0.0** beside the game executable or in the game root (or select it in the wizard).
3. Open a terminal in the extracted subfolder and run `./install.sh` — **not sudo**.
4. Follow the wizard, explicitly accept experimental reconstructed weight layouts, then use the printed launch wrapper in your launcher. In Steam, paste the printed launch options. Steam is optional.

The archive contains the prebuilt HIP library, Windows bridge, add-on, ReShade loader, matching modified vkd3d DLL pair and offline benchmark. No compiler is needed to use it. ROCm must already be installed; this installer does not install it.

## Requirements and limitations

- Linux x86_64, Python 3.10+, ROCm/HIP 7 and a compatible Wine/Proton runner. Binaries are built for **gfx1201**, not all AMD GPUs or distributions.
- An in-game DX12 FSR/FidelityFX hook. **No anti-cheat games, Magpie support or frame generation in this release.**
- The network operates at 1920×1080 internally; the live hook resizes other supported frame sizes. Motion history is reset for each live frame, so this is **not temporally complete**.
- Some weight layouts are reconstructed from AMD-consumer evidence. Opting in does not establish NVIDIA numerical or visual equivalence.
- The prototype uses CPU readback/upload and HIP execution at a split vkd3d submission boundary. **The game still waits for neural rendering.** It is not an asynchronous performance fix.
- General gameplay stability, HDR behavior and broad game compatibility are not certified. F6 toggles the live path's bypass when the hook is active; it cannot fix a missing hook or failed initialization.

A local release-preparation run on an RX 9070 XT (`gfx1201`), with a fixed 1080p input and real weights, measured **210–216 ms GPU time per inference** over three runs. That is network-only timing, **not in-game FPS**. Optimizing kernels, data transfers and memory use remains necessary.

## Troubleshooting and removal

If executable detection is ambiguous, see `./install.sh install --help` and use `--exe '/path/to/Game/Binaries/Win64/Game.exe'`. Keep the game's existing prefix and launch arguments. Do not reuse the old proprietary `version.dll` mod; this HIP wrapper selects Wine's builtin version library.

Logs: `DLSS5-AMD/logs/native-hip-live.txt` and `.dlssnr-linux/logs/`. Installation success means files were copied and verified, not that rendering works in your game. Stop testing if the game freezes or produces invalid output.

Uninstall: `./install.sh uninstall --exe /path/to/Game.exe --yes`, then remove the wrapper from the launcher. Managed original files are restored from backups. Preserve backups if integrity checks report changed files.

## Development and credits

[HIP backend / offline inference](hip/README.md) · [Build and source details](docs/BUILD.md) · [Third-party notices](linux/THIRD-PARTY.md)

Original D3D12/HLSL network and game integration: **Kien / lmxxf**. Linux HIP port, bridge and packaging: **guentra and AI collaborators**. The Windows upstream performance claims do not describe this port.

## Legal

Not affiliated with NVIDIA or AMD. No NVIDIA DLL or weights are distributed.
Use a copy of `nvngx_dlssnr.dll` you obtained legitimately, converted with the
lmxxf lab scripts, then packaged. Provided as-is.

Project code: [MIT](LICENSE). Modified vkd3d-proton: LGPL-2.1-or-later, with corresponding source provided alongside the binaries. Third-party components retain their own licenses.

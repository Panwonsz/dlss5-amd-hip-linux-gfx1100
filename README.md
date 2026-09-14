# DLSS5-AMD HIP for Linux — proof of concept

> **This is a slow, experimental proof of concept. It needs substantial optimization and is not ready for normal gameplay.** Expect very low frame rates, high latency and possible rendering problems. It is not an official NVIDIA DLSS implementation or a claim of equivalent image quality.

A Linux HIP/rocWMMA port of [lmxxf's DLSS5-AMD work](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting), with a ReShade add-on, Wine bridge and modified vkd3d-proton submission path. The complete 71-block network runs on AMD `gfx1201`. NVIDIA DLLs and weights are **not included**.

## Download and install

**[Download the prebuilt .tar.gz](https://github.com/guentra/dlss5-amd-hip-linux/releases/download/v0.2.0/dlss5-amd-hip-linux.tar.gz)** · [Release, checksums and corresponding source](https://github.com/guentra/dlss5-amd-hip-linux/releases/tag/v0.2.0)

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

The offline bench (network only, fixed 1080p input, real converted weights, RX 9070 XT `gfx1201`, warm runs) measures **~122 ms GPU time per inference** on the current build — the `0.1.0-poc` build measured 210–216 ms. That is network-only timing, **not in-game FPS**. Kernel work is tracked in the [Changelog](#changelog); data transfers and memory use remain open targets.

## Changelog

Unless a scenario is specified, timings are the offline bench (network only, fixed 1920×1080 input, real converted weights, RX 9070 XT `gfx1201`); "bit-exact" means the output did not change by a single bit against the reference 71-block network, which stays the judge.

| Version | Date | What changed | Result |
|---|---|---|---|
| `0.1.0-poc` | 09-13 | First complete Linux port: all 71 blocks as HIP/rocWMMA wave-matrix kernels on `gfx1201`; weights converted from the user's own `nvngx_dlssnr.dll` 310.8.0.0 (225 tables, resident in VRAM, no NVIDIA DLL at inference time); Win64→SysV trampoline + ReShade add-on + modified vkd3d-proton submission boundary for the in-game hook (synchronous, motion history reset per frame); offline `hip-network70` bench and `infer_image.py` | bench 210–216 ms, in-game staging only |
| `0.2.0` | 09-14 | lmxxf upstream sync 0.12 → 0.15 (≤1920×1080 window fit with aspect preservation, on-screen notice + bitmap font, FPS-display refresh, Windows-Update driver trap) and the Linux installer retargeted at the lmxxf ReShade add-on (wizard, managed backups, consent-based ROCm fallback, per-file sha256 manifest, install/trampoline/weights-safety test suites). GPU kernel optimization, all bit-exact (0 mismatches on every operator and graph test): fused QKV+LayerNorm+quant kernel (one dispatch per row group instead of GEMM → normalize → quant; the `ROWS_PER_THREAD` fix also removed 4× redundant row loads and out-of-bounds aliased stores); window-attention rewrite (padded score matrix kills the 32-way LDS bank conflict, softmax writes E4M3 bytes straight into the P·V operand with 16-byte stores, the provable-identity re-quantization stage is gone, 4 → 2 barriers); exact squares through opaque inline asm (`v_fma_mixlo_f16` / `v_mul_f32_e32`) instead of 32 volatile SCOPE_SYS private-memory round-trips per row-head | bench 175 → 167 → 131 → 122 ms (−30 % vs the pre-optimization build), bit-identical output; in-game verification in progress |
| `0.2.1` | 09-14 | Further GPU optimization, all bit-exact (operator + graph suites, network fingerprint unchanged): C32 FFN activation spill eliminated — `ActivatePolyC32`'s four volatile private-memory round-trips became opaque `v_mul_f32_e32` inline asm (2.0 → 0.45 ms/call, scratch 20 → 0 B, the kernel was occupancy-limited, not instruction-bound); the per-row-head squares in `k_normalize_qkv` use the same opaque asm helpers (scratch 40 → 36 B); LDS aliasing in `k_ffn_f32` and `k_qkv_norm_f32` (input/A-tile staging shares one LDS buffer, −512 B/block, C32 FFN occupancy 78 % → 100 %); `hip/tests/bench_ffn32.hip` isolated micro-benchmark + `make bench-ffn32`. ReShade add-on: the upscaler hook worker now also recognizes the self-contained FFX provider dlls (`amd_fidelityfx_upscaler_dx12.dll`, `amd_fidelityfx_framegeneration_dx12.dll`) that UE FSR-plugin titles load directly from `Engine/Plugins/Marketplace/FSR` signedbin — they export the same `ffxDispatch` API and the dispatch handler already filters on the upscaler header type; titles shipping only the 26 KB FFX loader next to the exe were unaffected | bench 122 → ≈98–105 ms (−15/20 %), bit-identical output; in-game FSR3/FSR4 takeover verified on Beast of Reincarnation (previously impossible: the hook waited for a loader dll that title never loads); Stellar Blade deployment refreshed with the same library |

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

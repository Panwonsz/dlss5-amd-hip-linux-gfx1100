# Third-party notices (Linux installer)

- Repository-root D3D12/HLSL add-on sources: MIT (see `/LICENSE`).
- ReShade 6.8.0 with full add-on support (`bin/ReShade64.dll`) is the official build from https://reshade.me/downloads/ReShade_Setup_6.8.0_Addon.exe (BSD-3-Clause; see `licenses/ReShade-LICENSE.md` and `licenses/ReShade-PROVENANCE.txt`). Its copyright, conditions and disclaimer are retained. This project is not endorsed by ReShade.
- MinHook is fetched by `scripts/build-addon-oneclick.sh` and linked into the add-on; its notice is included as `licenses/MinHook-LICENSE.txt` in the archive.
- Modified vkd3d-proton (`lmxxf-d3d12.dll`, `d3d12core.dll`): LGPL-2.1-or-later. See `licenses/vkd3d-LICENSE`, `licenses/vkd3d-COPYING` and `licenses/vkd3d-AUTHORS`. Complete corresponding source, dependencies and build instructions accompany the release as `vkd3d-proton-poc-source.tar.gz`. Local modifications include the local submission callback boundary, resource lifetime/ordering support and a unified-layout barrier correction; these are not upstream vkd3d-proton releases.
- NVIDIA DLSS weights and `nvngx_dlssnr.dll` are not included.
- Game/Proton discovery in `linux/dlssnr/games.py` comes from DLSS-NR-on-AMD-Linux `experimental/rx9060-gfx1200`.

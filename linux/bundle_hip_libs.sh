#!/bin/bash
# Make libdlss5_hip.so loadable inside Steam's Linux Runtime (pressure-vessel) container.
#
# Proton runs the game inside a container that has its own /usr, so host-only libraries
# (ROCm's libamdhip64 and its dependencies such as libfmt) are missing there. Because
# launch.sh puts libdlss5_hip.so in LD_PRELOAD, a missing dependency aborts every process
# Steam starts in the container ("libfmt.so.12: cannot open shared object file").
#
# This copies the library plus its non-glibc dependencies into one folder and sets
# RUNPATH=$ORIGIN on every copy, so the dependency chain resolves from that folder in any
# mount namespace. glibc, libstdc++ and libgcc_s are left to the container (pressure-vessel
# imports the newer host versions for the graphics stack).
#
# ldd is not the whole story. HIP loads the code object manager with dlopen when it creates a
# device, so libamd_comgr never appears as a dependency -- and without it ROCclr cannot build a
# device for an agent it can otherwise see:
#
#   comgrctx.cpp : Failed to load COMGR library.
#   rocdevice.cpp: Code object manager initialization failed for HSA device gfx1100
#   rocdevice.cpp: Error creating new instance of Device.
#
# which surfaces as hipSetDevice reporting "no ROCm-capable device is detected" while hsa_init
# succeeds and lists the GPU. On a normal system the loader finds it in /opt/rocm/lib; inside
# Steam's container it does not, because pressure-vessel rebuilds LD_LIBRARY_PATH. dlopen resolves
# against the calling library's RUNPATH, so a copy beside libamdhip64 is enough.
#
# Usage: bundle_hip_libs.sh /path/to/libdlss5_hip.so /path/to/output_lib_dir
set -euo pipefail

src=${1:?usage: bundle_hip_libs.sh LIBDLSS5_HIP_SO OUTPUT_DIR}
out=${2:?usage: bundle_hip_libs.sh LIBDLSS5_HIP_SO OUTPUT_DIR}
command -v patchelf >/dev/null || { echo "patchelf is required: sudo pacman -S patchelf" >&2; exit 1; }
[[ -f "$src" ]] || { echo "missing $src" >&2; exit 1; }
mkdir -p "$out"

skip_re='^(linux-vdso|ld-linux|libc\.so|libm\.so|libdl\.so|libpthread\.so|librt\.so|libresolv\.so|libutil\.so|libstdc\+\+\.so|libgcc_s\.so)'

src_real=$(readlink -f "$src")
out_real=$(readlink -f "$out")
if [[ "$(dirname "$src_real")" != "$out_real" ]]; then
    cp -f "$src_real" "$out/libdlss5_hip.so"
fi

count=0
while read -r name path; do
    [[ -n "$path" && -f "$path" ]] || continue
    [[ "$name" =~ $skip_re ]] && continue
    cp -fL "$path" "$out/$name"
    count=$((count + 1))
done < <(ldd "$src_real" | awk '/=>/ && $3 ~ /^\// {print $1, $3}')

# The dlopen-only libraries, by name, since nothing in the ELF headers points at them.
rocm_lib=$(dirname "$(ldd "$src_real" | awk '/libamdhip64/ && $3 ~ /^\// {print $3; exit}')")
[[ -d "$rocm_lib" ]] || rocm_lib=${ROCM_LIB:-/opt/rocm/lib}

for extra in "$rocm_lib"/libamd_comgr.so*; do
    [[ -e "$extra" ]] || continue
    name=$(basename "$extra")

    if [[ -L "$extra" ]]; then
        ln -sf "$(basename "$(readlink -f "$extra")")" "$out/$name"
        continue
    fi

    cp -f "$extra" "$out/$name"
    count=$((count + 1))

    # comgr brings its own dependencies, and they have to travel too.
    while read -r dep path; do
        [[ -n "$path" && -f "$path" ]] || continue
        [[ "$dep" =~ $skip_re ]] && continue
        [[ -f "$out/$dep" ]] && continue
        cp -fL "$path" "$out/$dep"
        count=$((count + 1))
    done < <(ldd "$extra" | awk '/=>/ && $3 ~ /^\// {print $1, $3}')
done

if ! ls "$out"/libamd_comgr.so* >/dev/null 2>&1; then
    echo "warning: no libamd_comgr found near $rocm_lib -- HIP will report no device inside the" >&2
    echo "         Steam runtime even though hsa_init succeeds. Set ROCM_LIB to point at it." >&2
fi

missing=$(ldd "$src_real" | grep "not found" || true)
if [[ -n "$missing" ]]; then
    echo "unresolved on the host (fix these first):" >&2
    echo "$missing" >&2
    exit 1
fi

for f in "$out"/*.so*; do
    chmod u+w "$f"
    patchelf --set-rpath '$ORIGIN' "$f"
done

echo "bundled libdlss5_hip.so + $count dependencies into $out"
echo "check (should list only libc/libm/libstdc++/libgcc_s/ld-linux outside $out):"
LD_LIBRARY_PATH= ldd "$out/libdlss5_hip.so" | grep -v "$out" || true

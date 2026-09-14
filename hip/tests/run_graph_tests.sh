#!/usr/bin/env bash
# Isolated native GPU tests. Never installs/launches a game or executes vendor code.
set -euo pipefail
cd "$(dirname "$0")/../.."
B=${DLSS5_GRAPH_TEST_BUILD:-linux/build/graph-audit}
export DLSS5_GRAPH_TEST_BUILD="$B"
HIPCC=/opt/rocm/bin/hipcc
mkdir -p "$B"
FLAGS=(--offload-arch=gfx1201 -O3 -std=c++17 -Ihip/include)
"$HIPCC" "${FLAGS[@]}" -c hip/src/kernels.hip -o "$B/kernels.o"
"$HIPCC" "${FLAGS[@]}" -c hip/src/network.hip -o "$B/network.o"
for t in test_wmma test_stages test_quantize test_attention test_residual test_linear test_logical_ops test_ffn_f32 test_graph_ops test_graph_blocks test_graph_stream test_graph_vit test_network test_configured_path; do
  precision=(-ffp-contract=off)
  # The precise C32 regression must run under production contraction settings.
  if [[ "$t" == test_configured_path ]]; then precision=(); fi
  "$HIPCC" "${FLAGS[@]}" "${precision[@]}" -c "hip/tests/$t.hip" -o "$B/$t.o"
  objects=("$B/$t.o" "$B/kernels.o")
  # Block and ViT tests include network.hip to reach the real internal scheduler.
  if [[ "$t" == test_network ]]; then objects+=("$B/network.o"); fi
  "$HIPCC" "${objects[@]}" -Wl,-rpath,/opt/rocm/lib -o "$B/$t"
  log="$B/$t.log"
  if [[ "$t" == test_network ]]; then log="$B/network-final.log"; fi
  timeout "${DLSS5_GRAPH_TEST_TIMEOUT:-300}" "$B/$t" > "$log" 2>&1
  printf '%s PASS\n' "$t"
done
python3 hip/tests/test_graph_evidence.py

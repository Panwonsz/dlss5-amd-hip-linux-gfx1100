HIPCC ?= /opt/rocm/bin/hipcc
ARCH ?= gfx1201
INCLUDES := -Iinclude -I/opt/rocm/include
LDFLAGS := -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64 -Wl,-rpath,/opt/rocm/lib
CXXFLAGS := -O3 -std=c++17 --offload-arch=$(ARCH) $(INCLUDES)
BUILD ?= build
TESTS := test_wmma test_stages test_quantize test_attention test_residual test_linear test_logical_ops test_ffn_f32

MINGW_CC ?= $(firstword $(wildcard \
  /home/guentra/dlssnr-linux-hip/toolchain/llvm-mingw/bin/x86_64-w64-mingw32-gcc \
  /home/guentra/dlssnr-linux-hip/toolchain/llvm-mingw-20260826-ucrt-ubuntu-22.04-x86_64/bin/x86_64-w64-mingw32-gcc \
  $(shell command -v x86_64-w64-mingw32-gcc 2>/dev/null)))

.PHONY: all clean test game test-api bench-ffn32 bench-qkv32 bench-block0 bench-attention

bench-ffn32: tests/bench_ffn32.hip $(BUILD)/kernels.o include/*.hpp
	$(HIPCC) $(CXXFLAGS) -c tests/bench_ffn32.hip -o $(BUILD)/bench_ffn32.o
	$(HIPCC) $(CXXFLAGS) $(BUILD)/bench_ffn32.o $(BUILD)/kernels.o $(LDFLAGS) -o $@

# Block 0's interval, kernel by kernel, at real sizes. ./bench-block0 [iters]
bench-block0: tests/bench_block0.hip $(BUILD)/kernels.o include/*.hpp
	$(HIPCC) $(CXXFLAGS) -c tests/bench_block0.hip -o $(BUILD)/bench_block0.o
	$(HIPCC) $(CXXFLAGS) $(BUILD)/bench_block0.o $(BUILD)/kernels.o $(LDFLAGS) -o $@

# k_window_attention at every width the network runs. ./bench-attention [iters]
bench-attention: tests/bench_attention.hip $(BUILD)/kernels.o include/*.hpp
	$(HIPCC) $(CXXFLAGS) -c tests/bench_attention.hip -o $(BUILD)/bench_attention.o
	$(HIPCC) $(CXXFLAGS) $(BUILD)/bench_attention.o $(BUILD)/kernels.o $(LDFLAGS) -o $@

# Standalone: it includes kernels/qkv_norm.hip itself and needs no kernels.o.
#   ./bench-qkv32 <tokens> <iters> <ah16> <w16> <nh>   -- nh=3 is what the network runs.
bench-qkv32: tests/bench_qkv32.hip kernels/qkv_norm.hip include/*.hpp
	mkdir -p $(BUILD)
	$(HIPCC) $(CXXFLAGS) -I. tests/bench_qkv32.hip $(LDFLAGS) -o $@

all: $(TESTS) hip-network70 libdlss5_hip.so

game: libdlss5_hip.so dlss5_hip.dll

test_wmma: tests/test_wmma.hip include/dlss5_linalg.hpp include/dlss5_common.hpp include/dlss5_runtime.hpp
	$(HIPCC) $(CXXFLAGS) tests/test_wmma.hip $(LDFLAGS) -o $@

$(BUILD)/kernels.o: src/kernels.hip include/*.hpp kernels/*.hip Makefile
	mkdir -p $(BUILD)
	$(HIPCC) $(CXXFLAGS) -fPIC -c src/kernels.hip -o $@

$(BUILD)/network.o: src/network.hip include/*.hpp Makefile
	mkdir -p $(BUILD)
	$(HIPCC) $(CXXFLAGS) -fPIC -c src/network.hip -o $@

$(BUILD)/%.o: src/%.hip include/*.hpp include/*.h Makefile
	mkdir -p $(BUILD)
	$(HIPCC) $(CXXFLAGS) -fPIC -c $< -o $@

$(filter-out test_wmma,$(TESTS)): %: tests/%.hip $(BUILD)/kernels.o include/*.hpp tests/ulp_compare.hpp
	$(HIPCC) $(CXXFLAGS) -c $< -o $(BUILD)/$@.o
	$(HIPCC) $(CXXFLAGS) $(BUILD)/$@.o $(BUILD)/kernels.o $(LDFLAGS) -o $@

hip-network70: $(BUILD)/bench.o $(BUILD)/kernels.o $(BUILD)/network.o
	$(HIPCC) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

libdlss5_hip.so: $(BUILD)/capi.o $(BUILD)/frame.o $(BUILD)/kernels.o $(BUILD)/network.o
	$(HIPCC) $(CXXFLAGS) -shared -fPIC $^ $(LDFLAGS) -o $@

dlss5_hip.dll: src/trampoline.c include/dlss5_capi.h
	$(MINGW_CC) -shared -O2 -o $@ src/trampoline.c

test: $(TESTS)
	@set -e; for t in $(TESTS); do ./$$t; done

test-api: libdlss5_hip.so
	$(HIPCC) $(CXXFLAGS) tests/test_buffer.hip $(LDFLAGS) -o $(BUILD)/test_buffer
	$(BUILD)/test_buffer
	$(HIPCC) $(CXXFLAGS) tests/test_capi.hip -L. -ldlss5_hip -Wl,-rpath,'$$ORIGIN/..' $(LDFLAGS) -o $(BUILD)/test_capi
	$(BUILD)/test_capi
	python3 tests/test_publish.py "$(CURDIR)/libdlss5_hip.so"

.PHONY: test-frame test-graph
test-frame: $(BUILD)/frame.o
	$(HIPCC) $(CXXFLAGS) -c tests/test_frame.hip -o $(BUILD)/test_frame.o
	$(HIPCC) $(BUILD)/test_frame.o $(BUILD)/frame.o $(LDFLAGS) -o $(BUILD)/test_frame
	$(BUILD)/test_frame
	$(HIPCC) $(CXXFLAGS) -c tests/test_temporal.hip -o $(BUILD)/test_temporal.o
	$(HIPCC) $(BUILD)/test_temporal.o $(BUILD)/frame.o $(LDFLAGS) -o $(BUILD)/test_temporal
	$(BUILD)/test_temporal

test-graph:
	bash tests/run_graph_tests.sh

.PHONY: check test-python test-model
# Run the GPU families serially even when the build uses -j.
check:
	$(MAKE) game hip-network70
	$(MAKE) test-api
	$(MAKE) test-frame
	$(MAKE) test-graph
	$(MAKE) test-python

test-python: hip-network70
	python3 tests/test_bench.py "$(CURDIR)/hip-network70"
	python3 tests/test_infer_image.py
	python3 tests/test_make_targets.py
	python3 -m unittest discover -s ../linux/tests -v

# Explicit real weights only: synthetic graph fixtures belong to test-graph.
test-model:
	@test -n "$(WEIGHTS)" || { printf '%s\n' 'Set WEIGHTS=/path/to/complete/tables'; exit 2; }
	$(MAKE) libdlss5_hip.so
	$(HIPCC) $(CXXFLAGS) -c tests/test_capi_lifecycle.hip -o $(BUILD)/test_capi_lifecycle.o
	$(HIPCC) $(BUILD)/test_capi_lifecycle.o -L. -ldlss5_hip -Wl,-rpath,"$(CURDIR)" $(LDFLAGS) -o $(BUILD)/test_capi_lifecycle
	$(BUILD)/test_capi_lifecycle "$(WEIGHTS)"
	$(HIPCC) $(CXXFLAGS) -c tests/test_frame_api.hip -o $(BUILD)/test_frame_api.o
	$(HIPCC) $(BUILD)/test_frame_api.o -L. -ldlss5_hip -Wl,-rpath,"$(CURDIR)" $(LDFLAGS) -o $(BUILD)/test_frame_api
	$(BUILD)/test_frame_api "$(WEIGHTS)"

clean:
	rm -f $(TESTS) hip-network70 libdlss5_hip.so dlss5_hip.dll $(BUILD)/*.o $(BUILD)/test_capi

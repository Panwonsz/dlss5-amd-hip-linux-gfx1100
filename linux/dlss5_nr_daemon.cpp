// DLSS-NR model server: runs the network in an ordinary host process and answers frames over loopback.
//
// Why this exists. The model used to run inside the game, through dlss5_hip.dll and a preloaded
// libdlss5_hip.so, and the game lost the GPU two seconds into the first frame. The reason first given
// here -- that a run with the recording switched off crashed too, so ROCm in the game process had to be
// the cause -- was wrong: that switch was broken and the run proved nothing. With the model out here,
// the game still dies at the same moment, so the cause is elsewhere and is still being hunted in
// DlssNr_Hip.cpp.
//
// The move is still right, though, for reasons that do not depend on that: 234 ms a frame is only
// reachable where ROCm is not fighting a Wine process for the GPU, the weights stay loaded across game
// launches, and this process is a normal Linux program on the host. It finds
// /opt/rocm and the weights cache by itself. Nothing needs bundling into the Steam container, no
// LD_PRELOAD, no libamd_comgr copy, no 601 MB of weights beside the game. The container stops being
// part of the problem.
//
// The protocol is deliberately dull: one frame per request, header then pixels, reply in the same
// shape. Loopback moves 33 MB in a couple of milliseconds, which is nothing against the model's ~230.
// The game sends the staged bytes exactly as D3D12 laid them out -- whatever format, whatever row
// pitch -- and this end does the conversion, because doing it here keeps it off the render thread and
// out of the code that has to survive inside a game process.
//
// Build:
//   g++ -O2 -std=c++17 -o dlss5-nr-daemon dlss5_nr_daemon.cpp -ldl -lpthread
//
// Run:
//   ./dlss5-nr-daemon --weights ~/.cache/dlss5-hip/weights/<schema>/<sha>
//
// It loads libdlss5_hip.so at runtime (--library to point elsewhere), so it does not need to be
// rebuilt when the model is.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <csignal>
#include <dlfcn.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

namespace
{

// ---------------------------------------------------------------------------------------------
// Protocol. Little-endian, fixed header, no negotiation: both ends ship together.

constexpr uint32_t kRequestMagic = 0x31524E44;  // "DNR1"
constexpr uint32_t kResponseMagic = 0x52524E44; // "DNRR"
constexpr uint16_t kDefaultPort = 47820;

struct Request
{
    uint32_t magic;
    uint32_t width;
    uint32_t height;
    uint32_t format;   // DXGI_FORMAT, as the game sees it
    uint32_t rowPitch; // bytes per row in the payload
    uint32_t bytes;    // payload size: rowPitch * height
    uint32_t reset;    // history reset, for when the model grows one
    uint32_t sequence;
};

struct Response
{
    uint32_t magic;
    uint32_t status; // 0 ok, anything else is a failure the game should report
    uint32_t bytes;  // payload size, same layout as the request
    uint32_t millis; // what the model took, so the game can show it
};

// DXGI format numbers, spelled out rather than included: this end never sees a Windows header.
enum : uint32_t
{
    kR32G32B32A32_FLOAT = 2,
    kR16G16B16A16_FLOAT = 10,
    kR10G10B10A2_UNORM = 24,
    kR11G11B10_FLOAT = 26,
    kR8G8B8A8_UNORM = 28,
    kR8G8B8A8_UNORM_SRGB = 29,
    kB8G8R8A8_UNORM = 87,
    kB8G8R8A8_UNORM_SRGB = 91,
};

constexpr uint32_t kModelWidth = 1920;
constexpr uint32_t kModelHeight = 1080;

// ---------------------------------------------------------------------------------------------
// Pixel conversion, between the frame's own format and the packed float the model reads.

float HalfToFloat(uint16_t h)
{
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1Fu;
    uint32_t mantissa = h & 0x3FFu;
    uint32_t bits;

    if (exponent == 0)
    {
        if (mantissa == 0)
        {
            bits = sign;
            float out;
            std::memcpy(&out, &bits, 4);
            return out;
        }

        while ((mantissa & 0x400u) == 0)
        {
            mantissa <<= 1;
            exponent--;
        }

        exponent++;
        mantissa &= 0x3FFu;
    }
    else if (exponent == 31)
    {
        bits = sign | 0x7F800000u | (mantissa << 13);
        float out;
        std::memcpy(&out, &bits, 4);
        return out;
    }

    bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

uint16_t FloatToHalf(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, 4);

    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t rawExponent = (bits >> 23) & 0xFFu;
    const uint32_t mantissa = bits & 0x7FFFFFu;

    if (rawExponent == 0xFFu)
        return uint16_t(sign | 0x7C00u | (mantissa != 0 ? 0x200u : 0u));

    const int32_t exponent = int32_t(rawExponent) - 112;

    if (exponent >= 0x1F)
        return uint16_t(sign | 0x7BFFu); // saturate rather than hand the frame an infinity

    if (exponent <= 0)
    {
        if (exponent < -10)
            return uint16_t(sign);

        const uint32_t withImplied = mantissa | 0x800000u;
        const uint32_t shift = uint32_t(14 - exponent);
        const uint32_t truncated = withImplied >> shift;
        const uint32_t half = 1u << (shift - 1);
        const uint32_t remainder = withImplied & ((1u << shift) - 1u);
        return uint16_t(sign + truncated +
                        ((remainder > half || (remainder == half && (truncated & 1u) != 0)) ? 1u : 0u));
    }

    const uint32_t packed = (uint32_t(exponent) << 10) | (mantissa >> 13);
    const uint32_t remainder = mantissa & 0x1FFFu;
    const uint32_t rounded =
        packed + ((remainder > 0x1000u || (remainder == 0x1000u && (packed & 1u) != 0)) ? 1u : 0u);
    return uint16_t(sign | (rounded >= 0x7C00u ? 0x7BFFu : rounded));
}

float Float11ToFloat(uint32_t v)
{
    const uint32_t exponent = (v >> 6) & 0x1Fu;
    const uint32_t mantissa = v & 0x3Fu;
    const uint32_t bits = exponent == 0 ? 0u : ((exponent + 112u) << 23) | (mantissa << 17);
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

uint32_t FloatToFloat11(float f)
{
    if (!(f > 0.0f))
        return 0;

    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const int32_t exponent = int32_t((bits >> 23) & 0xFFu) - 112;

    if (exponent <= 0)
        return 0;

    if (exponent >= 0x1F)
        return 0x7BFu;

    return (uint32_t(exponent) << 6) | ((bits >> 17) & 0x3Fu);
}

float Float10ToFloat(uint32_t v)
{
    const uint32_t exponent = (v >> 5) & 0x1Fu;
    const uint32_t mantissa = v & 0x1Fu;
    const uint32_t bits = exponent == 0 ? 0u : ((exponent + 112u) << 23) | (mantissa << 18);
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

uint32_t FloatToFloat10(float f)
{
    if (!(f > 0.0f))
        return 0;

    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const int32_t exponent = int32_t((bits >> 23) & 0xFFu) - 112;

    if (exponent <= 0)
        return 0;

    if (exponent >= 0x1F)
        return 0x3DFu;

    return (uint32_t(exponent) << 5) | ((bits >> 18) & 0x1Fu);
}

float Saturate(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

bool FormatSupported(uint32_t format)
{
    switch (format)
    {
    case kR32G32B32A32_FLOAT:
    case kR16G16B16A16_FLOAT:
    case kR10G10B10A2_UNORM:
    case kR11G11B10_FLOAT:
    case kR8G8B8A8_UNORM:
    case kR8G8B8A8_UNORM_SRGB:
    case kB8G8R8A8_UNORM:
    case kB8G8R8A8_UNORM_SRGB:
        return true;
    default:
        return false;
    }
}

void DecodeRow(const uint8_t* src, float* dst, uint32_t format, uint32_t width)
{
    switch (format)
    {
    case kR16G16B16A16_FLOAT:
    {
        const auto* p = reinterpret_cast<const uint16_t*>(src);

        for (uint32_t x = 0; x < width; x++)
        {
            dst[x * 4 + 0] = HalfToFloat(p[x * 4 + 0]);
            dst[x * 4 + 1] = HalfToFloat(p[x * 4 + 1]);
            dst[x * 4 + 2] = HalfToFloat(p[x * 4 + 2]);
            dst[x * 4 + 3] = 1.0f;
        }

        break;
    }
    case kR32G32B32A32_FLOAT:
    {
        const auto* p = reinterpret_cast<const float*>(src);

        for (uint32_t x = 0; x < width; x++)
        {
            dst[x * 4 + 0] = p[x * 4 + 0];
            dst[x * 4 + 1] = p[x * 4 + 1];
            dst[x * 4 + 2] = p[x * 4 + 2];
            dst[x * 4 + 3] = 1.0f;
        }

        break;
    }
    case kR8G8B8A8_UNORM:
    case kR8G8B8A8_UNORM_SRGB:
    case kB8G8R8A8_UNORM:
    case kB8G8R8A8_UNORM_SRGB:
    {
        const bool bgr = format == kB8G8R8A8_UNORM || format == kB8G8R8A8_UNORM_SRGB;

        for (uint32_t x = 0; x < width; x++)
        {
            dst[x * 4 + 0] = float(src[x * 4 + (bgr ? 2 : 0)]) / 255.0f;
            dst[x * 4 + 1] = float(src[x * 4 + 1]) / 255.0f;
            dst[x * 4 + 2] = float(src[x * 4 + (bgr ? 0 : 2)]) / 255.0f;
            dst[x * 4 + 3] = 1.0f;
        }

        break;
    }
    case kR10G10B10A2_UNORM:
    {
        const auto* p = reinterpret_cast<const uint32_t*>(src);

        for (uint32_t x = 0; x < width; x++)
        {
            const uint32_t v = p[x];
            dst[x * 4 + 0] = float(v & 0x3FFu) / 1023.0f;
            dst[x * 4 + 1] = float((v >> 10) & 0x3FFu) / 1023.0f;
            dst[x * 4 + 2] = float((v >> 20) & 0x3FFu) / 1023.0f;
            dst[x * 4 + 3] = 1.0f;
        }

        break;
    }
    case kR11G11B10_FLOAT:
    {
        const auto* p = reinterpret_cast<const uint32_t*>(src);

        for (uint32_t x = 0; x < width; x++)
        {
            const uint32_t v = p[x];
            dst[x * 4 + 0] = Float11ToFloat(v & 0x7FFu);
            dst[x * 4 + 1] = Float11ToFloat((v >> 11) & 0x7FFu);
            dst[x * 4 + 2] = Float10ToFloat((v >> 22) & 0x3FFu);
            dst[x * 4 + 3] = 1.0f;
        }

        break;
    }
    default:
        break;
    }
}

void EncodeRow(const float* src, uint8_t* dst, uint32_t format, uint32_t width)
{
    switch (format)
    {
    case kR16G16B16A16_FLOAT:
    {
        auto* p = reinterpret_cast<uint16_t*>(dst);

        for (uint32_t x = 0; x < width; x++)
        {
            p[x * 4 + 0] = FloatToHalf(src[x * 3 + 0]);
            p[x * 4 + 1] = FloatToHalf(src[x * 3 + 1]);
            p[x * 4 + 2] = FloatToHalf(src[x * 3 + 2]);
            p[x * 4 + 3] = FloatToHalf(1.0f);
        }

        break;
    }
    case kR32G32B32A32_FLOAT:
    {
        auto* p = reinterpret_cast<float*>(dst);

        for (uint32_t x = 0; x < width; x++)
        {
            p[x * 4 + 0] = src[x * 3 + 0];
            p[x * 4 + 1] = src[x * 3 + 1];
            p[x * 4 + 2] = src[x * 3 + 2];
            p[x * 4 + 3] = 1.0f;
        }

        break;
    }
    case kR8G8B8A8_UNORM:
    case kR8G8B8A8_UNORM_SRGB:
    case kB8G8R8A8_UNORM:
    case kB8G8R8A8_UNORM_SRGB:
    {
        const bool bgr = format == kB8G8R8A8_UNORM || format == kB8G8R8A8_UNORM_SRGB;

        for (uint32_t x = 0; x < width; x++)
        {
            const auto r = uint8_t(Saturate(src[x * 3 + 0]) * 255.0f + 0.5f);
            const auto g = uint8_t(Saturate(src[x * 3 + 1]) * 255.0f + 0.5f);
            const auto b = uint8_t(Saturate(src[x * 3 + 2]) * 255.0f + 0.5f);
            dst[x * 4 + (bgr ? 2 : 0)] = r;
            dst[x * 4 + 1] = g;
            dst[x * 4 + (bgr ? 0 : 2)] = b;
            dst[x * 4 + 3] = 255;
        }

        break;
    }
    case kR10G10B10A2_UNORM:
    {
        auto* p = reinterpret_cast<uint32_t*>(dst);

        for (uint32_t x = 0; x < width; x++)
        {
            const auto r = uint32_t(Saturate(src[x * 3 + 0]) * 1023.0f + 0.5f);
            const auto g = uint32_t(Saturate(src[x * 3 + 1]) * 1023.0f + 0.5f);
            const auto b = uint32_t(Saturate(src[x * 3 + 2]) * 1023.0f + 0.5f);
            p[x] = r | (g << 10) | (b << 20) | (3u << 30);
        }

        break;
    }
    case kR11G11B10_FLOAT:
    {
        auto* p = reinterpret_cast<uint32_t*>(dst);

        for (uint32_t x = 0; x < width; x++)
            p[x] = FloatToFloat11(src[x * 3 + 0]) | (FloatToFloat11(src[x * 3 + 1]) << 11) |
                   (FloatToFloat10(src[x * 3 + 2]) << 22);

        break;
    }
    default:
        break;
    }
}

// ---------------------------------------------------------------------------------------------
// The model, loaded at runtime so this binary does not have to be rebuilt when the network is.

using PFN_Init = int (*)(const char*, int);
using PFN_Run = int (*)(const float*, float*, unsigned int);
using PFN_LastError = const char* (*)();
using PFN_Shutdown = void (*)();

struct Model
{
    void* handle = nullptr;
    PFN_Init init = nullptr;
    PFN_Run run = nullptr;
    PFN_LastError lastError = nullptr;
    PFN_Shutdown shutdown = nullptr;
    const char* (*formats)() = nullptr;

    bool Load(const std::string& path)
    {
        handle = dlopen(path.c_str(), RTLD_NOW);

        if (handle == nullptr)
        {
            std::fprintf(stderr, "cannot load %s: %s\n", path.c_str(), dlerror());
            return false;
        }

        init = (PFN_Init) dlsym(handle, "dlss5_init");
        run = (PFN_Run) dlsym(handle, "dlss5_run");
        lastError = (PFN_LastError) dlsym(handle, "dlss5_last_error");
        shutdown = (PFN_Shutdown) dlsym(handle, "dlss5_shutdown");
        // Optional: absent in libraries built before 2026-09-21, and not worth failing over.
        formats = (const char* (*)()) dlsym(handle, "dlss5_formats");

        if (init == nullptr || run == nullptr)
        {
            std::fprintf(stderr, "%s is missing dlss5_init/dlss5_run\n", path.c_str());
            return false;
        }

        return true;
    }

    const char* Error() const { return lastError != nullptr ? lastError() : "unknown"; }
    const char* Formats() const { return formats != nullptr ? formats() : "unknown (library predates dlss5_formats)"; }
};

std::atomic<bool> g_stop { false };

void OnSignal(int) { g_stop = true; }

// Converting two million pixels twice costs ~35 ms single-threaded, which is a seventh of the
// model's own time for work that is embarrassingly parallel and touches nothing shared. Split by
// rows across the cores that are sitting idle while the GPU works.
void ForEachRowBand(uint32_t height, const std::function<void(uint32_t, uint32_t)>& body)
{
    unsigned int bands = std::thread::hardware_concurrency();
    bands = bands < 2 ? 1 : (bands > 8 ? 8 : bands);

    if (bands == 1)
    {
        body(0, height);
        return;
    }

    const uint32_t per = (height + bands - 1) / bands;
    std::vector<std::thread> workers;
    workers.reserve(bands);

    for (unsigned int i = 0; i < bands; i++)
    {
        const uint32_t from = uint32_t(i) * per;

        if (from >= height)
            break;

        const uint32_t to = std::min(from + per, height);
        workers.emplace_back([&body, from, to] { body(from, to); });
    }

    for (auto& worker : workers)
        worker.join();
}

bool ReadExactly(int fd, void* into, size_t bytes)
{
    auto* p = static_cast<uint8_t*>(into);

    while (bytes > 0)
    {
        const ssize_t got = recv(fd, p, bytes, 0);

        if (got <= 0)
            return false;

        p += got;
        bytes -= size_t(got);
    }

    return true;
}

bool WriteExactly(int fd, const void* from, size_t bytes)
{
    const auto* p = static_cast<const uint8_t*>(from);

    while (bytes > 0)
    {
        const ssize_t put = send(fd, p, bytes, MSG_NOSIGNAL);

        if (put <= 0)
            return false;

        p += put;
        bytes -= size_t(put);
    }

    return true;
}

void Serve(int client, Model& model, bool verbose)
{
    // One frame's worth of everything, grown once and then reused.
    std::vector<uint8_t> staged;
    std::vector<float> input(size_t(kModelWidth) * kModelHeight * 4);
    std::vector<float> answer(size_t(kModelWidth) * kModelHeight * 3);
    uint64_t frames = 0;

    for (;;)
    {
        Request request {};

        if (!ReadExactly(client, &request, sizeof request))
            break; // the game closed the connection, which is how a session normally ends

        if (request.magic != kRequestMagic)
        {
            std::fprintf(stderr, "bad magic 0x%08X, dropping this client\n", request.magic);
            break;
        }

        if (request.width != kModelWidth || request.height != kModelHeight ||
            !FormatSupported(request.format) || request.bytes != request.rowPitch * request.height ||
            request.rowPitch < request.width * 4 || request.bytes > (256u << 20))
        {
            std::fprintf(stderr, "refusing frame %ux%u format %u pitch %u bytes %u\n", request.width,
                         request.height, request.format, request.rowPitch, request.bytes);
            Response bad { kResponseMagic, 2, 0, 0 };
            WriteExactly(client, &bad, sizeof bad);
            continue;
        }

        staged.resize(request.bytes);

        if (!ReadExactly(client, staged.data(), staged.size()))
            break;

        const auto started = std::chrono::steady_clock::now();

        ForEachRowBand(request.height,
                       [&](uint32_t from, uint32_t to)
                       {
                           for (uint32_t y = from; y < to; y++)
                               DecodeRow(staged.data() + size_t(y) * request.rowPitch,
                                         input.data() + size_t(y) * request.width * 4, request.format,
                                         request.width);
                       });

        const int rc = model.run(input.data(), answer.data(), uint32_t(frames));

        if (rc != 0)
        {
            std::fprintf(stderr, "model failed: %s\n", model.Error());
            Response bad { kResponseMagic, 1, 0, 0 };

            if (!WriteExactly(client, &bad, sizeof bad))
                break;

            continue;
        }

        ForEachRowBand(request.height,
                       [&](uint32_t from, uint32_t to)
                       {
                           for (uint32_t y = from; y < to; y++)
                               EncodeRow(answer.data() + size_t(y) * request.width * 3,
                                         staged.data() + size_t(y) * request.rowPitch, request.format,
                                         request.width);
                       });

        const auto millis = uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(
                                         std::chrono::steady_clock::now() - started)
                                         .count());

        Response ok { kResponseMagic, 0, request.bytes, millis };

        if (!WriteExactly(client, &ok, sizeof ok) || !WriteExactly(client, staged.data(), staged.size()))
            break;

        frames++;

        if (verbose || frames <= 3 || frames % 100 == 0)
            std::printf("frame %llu  %ux%u format %u  %u ms\n", (unsigned long long) frames,
                        request.width, request.height, request.format, millis);

        std::fflush(stdout);
    }

    std::printf("client gone after %llu frame(s)\n", (unsigned long long) frames);
    std::fflush(stdout);
}

std::string HomePath(const char* tail)
{
    const char* home = getenv("HOME");
    return std::string(home != nullptr ? home : ".") + tail;
}

} // namespace

int main(int argc, char** argv)
{
    std::string weights;
    std::string library = HomePath("/Downloads/AUR/dlss5-amd-hip-linux/hip/libdlss5_hip.so");
    uint16_t port = kDefaultPort;
    int gpu = 0;
    bool verbose = false;

    for (int i = 1; i < argc; i++)
    {
        const std::string arg = argv[i];
        const bool hasNext = i + 1 < argc;

        if (arg == "--weights" && hasNext)
            weights = argv[++i];
        else if (arg == "--library" && hasNext)
            library = argv[++i];
        else if (arg == "--port" && hasNext)
            port = uint16_t(std::atoi(argv[++i]));
        else if (arg == "--gpu" && hasNext)
            gpu = std::atoi(argv[++i]);
        else if (arg == "--verbose")
            verbose = true;
        else
        {
            std::fprintf(stderr,
                         "usage: %s --weights DIR [--library libdlss5_hip.so] [--port %u] [--gpu 0]"
                         " [--verbose]\n",
                         argv[0], kDefaultPort);
            return 2;
        }
    }

    if (weights.empty())
    {
        std::fprintf(stderr, "--weights is required (the folder containing manifest.json)\n");
        return 2;
    }

    // The port comes first, before the model. Claiming it costs milliseconds and loading 148 MB of
    // weights costs seconds, so a port that is already taken should be found out immediately rather
    // than after a long wait -- which is exactly how it was found out the first time.
    //
    // Loopback only: this speaks to a game on the same machine and has no business being reachable
    // from anywhere else.
    const int listener = socket(AF_INET, SOCK_STREAM, 0);

    if (listener < 0)
    {
        std::perror("socket");
        return 1;
    }

    int reuse = 1;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof reuse);

    sockaddr_in address {};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(port);

    if (bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof address) != 0)
    {
        std::perror("bind");
        std::fprintf(stderr, "another daemon is probably already on port %u -- dlss5-nr-run.sh starts\n"
                             "one that outlives the game on purpose. Find it with:\n"
                             "    ss -ltnp | grep %u\n"
                             "and either use it as it is (the weights are already loaded) or stop it with:\n"
                             "    kill $(pgrep -x dlss5-nr-daemon)\n",
                     port, port);
        return 1;
    }

    listen(listener, 4);
    std::printf("listening on 127.0.0.1:%u\n", port);
    std::fflush(stdout);

    // sigaction rather than signal, and deliberately without SA_RESTART: glibc's signal() restarts
    // interrupted calls, so a Ctrl+C at an idle daemon was swallowed by accept() and only took effect
    // after the next game connected. Without the flag accept() returns EINTR and the loop ends.
    struct sigaction stop {};
    stop.sa_handler = OnSignal;
    sigemptyset(&stop.sa_mask);
    stop.sa_flags = 0;
    sigaction(SIGINT, &stop, nullptr);
    sigaction(SIGTERM, &stop, nullptr);
    Model model;

    if (!model.Load(library))
        return 1;

    std::printf("loading the model from %s\n", weights.c_str());
    std::fflush(stdout);

    if (model.init(weights.c_str(), gpu) != 0)
    {
        std::fprintf(stderr, "the model would not initialise: %s\n", model.Error());
        return 1;
    }

    // The daemon outlives the game and the next launch reuses it, so a DLSS5_W16 set in a Steam
    // launch option reaches a process that may have been started hours ago with the other value.
    // Print what this process actually loaded; the log is then evidence rather than intent.
    std::printf("model ready, %ux%u %s\n", kModelWidth, kModelHeight, model.Formats());

    std::signal(SIGPIPE, SIG_IGN);

    while (!g_stop)
    {
        const int client = accept(listener, nullptr, nullptr);

        if (client < 0)
        {
            if (g_stop)
                break;

            continue;
        }

        // Frames are big and latency matters more than packing, so no Nagle.
        int nodelay = 1;
        setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof nodelay);

        std::printf("game connected\n");
        std::fflush(stdout);
        Serve(client, model, verbose);
        close(client);
    }

    close(listener);

    if (model.shutdown != nullptr)
        model.shutdown();

    std::printf("stopped\n");
    return 0;
}

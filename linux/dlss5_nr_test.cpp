// Speaks the same wire as the in-game backend, so the daemon gets exercised the way the game will.
#include <arpa/inet.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>
#include <chrono>

struct Request { uint32_t magic, width, height, format, rowPitch, bytes, reset, sequence; };
struct Response { uint32_t magic, status, bytes, millis; };

static uint16_t FloatToHalf(float f) {                      // good enough for a test pattern
    uint32_t b; std::memcpy(&b, &f, 4);
    uint32_t sign = (b >> 16) & 0x8000u; int e = int((b >> 23) & 0xFF) - 112; uint32_t m = b & 0x7FFFFF;
    if (e <= 0) return uint16_t(sign);
    if (e >= 31) return uint16_t(sign | 0x7BFF);
    return uint16_t(sign | (uint32_t(e) << 10) | (m >> 13));
}
static float HalfToFloat(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000u) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    if (e == 0) { float z = 0.0f; uint32_t zb = sign; std::memcpy(&z, &zb, 4); return z; }
    uint32_t b = sign | ((e + 112u) << 23) | (m << 13); float o; std::memcpy(&o, &b, 4); return o;
}
static bool All(int fd, void* p, size_t n, bool recving) {
    auto* c = static_cast<char*>(p);
    while (n) { ssize_t k = recving ? recv(fd, c, n, 0) : send(fd, c, n, 0); if (k <= 0) return false; c += k; n -= size_t(k); }
    return true;
}

// Sends frames to dlss5-nr-daemon the way the game will, and reports what comes back.
//   ./dlss5-nr-test [port] [--stub]
// --stub also checks the exact transform of the fake model used in testing; without it the check is
// the one that matters against the real network: finite, in range, not uniformly black.
int main(int argc, char** argv) {
    const uint32_t W = 1920, H = 1080;
    const uint32_t pitch = 1920 * 8 + 256;   // deliberately padded, the way GetCopyableFootprints pads
    const int port = argc > 1 ? atoi(argv[1]) : 47820;

    int s = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a {}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = htons(uint16_t(port));
    if (connect(s, (sockaddr*) &a, sizeof a) != 0) { std::perror("connect"); return 1; }
    int one = 1; setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);

    std::vector<uint8_t> frame(size_t(pitch) * H, 0xAB);   // padding filled with junk on purpose
    for (uint32_t y = 0; y < H; y++) {
        auto* row = reinterpret_cast<uint16_t*>(frame.data() + size_t(y) * pitch);
        for (uint32_t x = 0; x < W; x++) {
            row[x * 4 + 0] = FloatToHalf(float(x % 256) / 255.0f);
            row[x * 4 + 1] = FloatToHalf(float(y % 256) / 255.0f);
            row[x * 4 + 2] = FloatToHalf(0.25f);
            row[x * 4 + 3] = FloatToHalf(1.0f);
        }
    }

    int bad = 0;
    const bool expectStub = argc > 2 && std::strcmp(argv[2], "--stub") == 0;

    for (int f = 0; f < 3; f++) {
        Request r { 0x31524E44, W, H, 10 /* R16G16B16A16_FLOAT */, pitch, uint32_t(pitch * H), 0, uint32_t(f) };
        auto t0 = std::chrono::steady_clock::now();
        if (!All(s, &r, sizeof r, false) || !All(s, frame.data(), frame.size(), false)) { std::puts("send failed"); return 1; }
        Response resp {};
        if (!All(s, &resp, sizeof resp, true)) { std::puts("no response"); return 1; }
        if (resp.magic != 0x52524E44 || resp.status != 0 || resp.bytes != pitch * H) {
            std::printf("bad response: magic %08X status %u bytes %u\n", resp.magic, resp.status, resp.bytes); return 1;
        }
        std::vector<uint8_t> back(resp.bytes);
        if (!All(s, back.data(), back.size(), true)) { std::puts("truncated payload"); return 1; }
        auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        // What the answer looks like. Against the real network this is the whole test: a frame that
        // is finite, inside a sane range and not uniformly one value is a frame the model produced.
        double lo = 1e30, hi = -1e30, sum = 0.0; size_t n = 0, bogus = 0, zero = 0;
        for (uint32_t y = 0; y < H; y += 7) {
            auto* row = reinterpret_cast<const uint16_t*>(back.data() + size_t(y) * pitch);
            for (uint32_t x = 0; x < W; x += 5) {
                for (int c = 0; c < 3; c++) {
                    float v = HalfToFloat(row[x * 4 + c]);
                    if (!(v == v) || v > 1e4f || v < -1e4f) { bogus++; continue; }
                    if (v == 0.0f) zero++;
                    lo = v < lo ? v : lo; hi = v > hi ? v : hi; sum += v; n++;
                    if (expectStub) {
                        float want[3] = { float(x % 256) / 255.0f * 0.5f, float(y % 256) / 255.0f * 0.5f, 0.125f };
                        if (v < want[c] - 0.002f || v > want[c] + 0.002f) {
                            if (++bad < 5) std::printf("  mismatch at %u,%u c%d: got %.4f want %.4f\n", x, y, c, v, want[c]);
                        }
                    }
                }
            }
        }

        std::printf("frame %d: %u bytes back, model %u ms, round trip %.0f ms | range %.4f..%.4f mean %.4f",
                    f, resp.bytes, resp.millis, ms, lo, hi, n ? sum / double(n) : 0.0);
        if (bogus) std::printf("  | %zu NaN/absurd values", bogus);
        if (n && zero == n) std::printf("  | ALL BLACK");
        std::puts("");
        if (bogus) bad++;
        if (n && zero == n) bad++;
    }

    close(s);
    if (bad) std::printf("FAILED: %d problem(s) above\n", bad);
    else std::puts(expectStub ? "all sampled pixels correct" :
                   "three frames came back finite and non-black -- the model is answering");
    return bad ? 1 : 0;
}

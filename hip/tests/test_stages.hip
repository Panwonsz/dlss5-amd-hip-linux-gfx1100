#include "dlss5_launch.hpp"
#include "dlss5_runtime.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace dlss5;

static int g_fail;

static void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        g_fail++;
    } else
        std::printf("ok   %s\n", what);
}

static u8 q8(float v) { return e4m3_byte(v); }

int main() {
    hip_check(hipSetDevice(0), "setDevice");
    hipDeviceProp_t prop{};
    hip_check(hipGetDeviceProperties(&prop, 0), "props");
    std::printf("device=%s\n", prop.name);

    // learned DS 16×16 C32→C64
    {
        const uint w = 16, h = 16, cin = 32, cout = 64;
        std::vector<u8> in(w * h * cin, q8(1.f)), wt(cout * cin, q8(0.f)), out(w / 2 * h / 2 * cout);
        for (uint oc = 0; oc < cout; oc++)
            wt[oc * cin + (oc % cin)] = q8(1.f);
        DeviceBuf d_in(in.size()), d_w(wt.size()), d_out(out.size());
        d_in.upload(in.data(), in.size());
        d_w.upload(wt.data(), wt.size());
        launch_ds(d_in.as<u8>(), d_w.as<u8>(), d_out.as<u8>(), w, h, cin, cout);
        hip_check(hipDeviceSynchronize(), "ds");
        d_out.download(out.data(), out.size());
        int nz = 0, bad = 0;
        for (auto v : out) {
            float f = from_e4m3(v);
            if (!std::isfinite(f))
                bad++;
            if (fabsf(f) > 1e-6f)
                nz++;
        }
        expect(bad == 0 && nz > 0, "launch_ds finite-nonzero");
        std::vector<u8> cpu(out.size());
        for (uint y = 0; y < h / 2; y++)
            for (uint x = 0; x < w / 2; x++)
                for (uint oc = 0; oc < cout; oc++) {
                    float acc = 0.f;
                    uint c = oc % cin;
                    float s = 0.f;
                    for (int dy = 0; dy < 2; dy++)
                        for (int dx = 0; dx < 2; dx++)
                            s += from_e4m3(in[((y * 2 + dy) * w + (x * 2 + dx)) * cin + c]);
                    acc = (s * 0.25f) * from_e4m3(wt[oc * cin + c]);
                    cpu[(y * (w / 2) + x) * cout + oc] = e4m3_byte(H(acc));
                }
        int mismatch = 0;
        for (size_t i = 0; i < out.size(); i++)
            if (out[i] != cpu[i])
                mismatch++;
        expect(mismatch == 0, "launch_ds cpu-match");
    }

    // C512 split ffwd, 16 tokens
    {
        const uint tokens = 16;
        std::vector<u8> in(tokens * 512, q8(0.5f));
        std::vector<float> pre(512 * 512, 0.f), ex(8 * 256 * 64, 0.f), ct(8 * 64 * 256, 0.f);
        for (int i = 0; i < 512; i++)
            pre[i * 512 + i] = 1.f;
        for (int g = 0; g < 8; g++)
            for (int i = 0; i < 64; i++) {
                ex[g * 256 * 64 + i * 64 + i] = 1.f;
                ct[g * 64 * 256 + i * 256 + i] = 1.f;
            }
        std::vector<u8> pre_t(512 * 512), ex_t(8 * 256 * 64), ct_t(8 * 64 * 256), out(tokens * 512);
        pack_tiled_e4m3(pre_t.data(), pre.data(), 512, 512);
        for (int g = 0; g < 8; g++) {
            pack_tiled_e4m3(ex_t.data() + g * 256 * 64, ex.data() + g * 256 * 64, 256, 64);
            pack_tiled_e4m3(ct_t.data() + g * 64 * 256, ct.data() + g * 64 * 256, 64, 256);
        }
        DeviceBuf d_in(in.size()), d_pre(pre_t.size()), d_ex(ex_t.size()), d_ct(ct_t.size()),
            d_out(out.size());
        d_in.upload(in.data(), in.size());
        d_pre.upload(pre_t.data(), pre_t.size());
        d_ex.upload(ex_t.data(), ex_t.size());
        d_ct.upload(ct_t.data(), ct_t.size());
        launch_split_ffwd(d_in.as<u8>(), d_pre.as<u8>(), d_ex.as<u8>(), d_ct.as<u8>(), d_out.as<u8>(),
                          tokens);
        hip_check(hipDeviceSynchronize(), "split");
        d_out.download(out.data(), out.size());
        int nz = 0, bad = 0;
        for (auto v : out) {
            float f = from_e4m3(v);
            if (!std::isfinite(f))
                bad++;
            if (fabsf(f) > 1e-6f)
                nz++;
        }
        expect(bad == 0 && nz > 0, "launch_split_ffwd finite-nonzero");
    }

    // ViT attention 16 tokens × 32 dim
    {
        const uint tokens = 16, dim = 32;
        std::vector<float> qkv(tokens * 3 * dim, 0.f);
        for (uint t = 0; t < tokens; t++)
            for (uint d = 0; d < dim; d++) {
                qkv[t * 3 * dim + d] = 0.125f;
                qkv[t * 3 * dim + dim + d] = 0.125f;
                qkv[t * 3 * dim + 2 * dim + d] = 0.25f;
            }
        std::vector<u8> out(tokens * dim);
        DeviceBuf d_q(qkv.size() * 4), d_o(out.size());
        d_q.upload(qkv.data(), qkv.size() * 4);
        launch_vit_attention(d_q.as<float>(), d_o.as<u8>(), tokens, dim);
        hip_check(hipDeviceSynchronize(), "vit");
        d_o.download(out.data(), out.size());
        int nz = 0, bad = 0;
        for (auto v : out) {
            float f = from_e4m3(v);
            if (!std::isfinite(f))
                bad++;
            if (fabsf(f) > 1e-6f)
                nz++;
        }
        expect(bad == 0 && nz > 0, "launch_vit_attention finite-nonzero");
    }

    // decoder upsample-add 8×8 C32
    {
        const uint w = 8, h = 8, c = 32;
        std::vector<u8> lo((w / 2) * (h / 2) * c, q8(0.5f)), skip(w * h * c, q8(0.25f)), out(w * h * c);
        DeviceBuf d_lo(lo.size()), d_sk(skip.size()), d_o(out.size());
        d_lo.upload(lo.data(), lo.size());
        d_sk.upload(skip.data(), skip.size());
        launch_upsample_add(d_lo.as<u8>(), d_sk.as<u8>(), d_o.as<u8>(), w / 2, h / 2, c);
        hip_check(hipDeviceSynchronize(), "up");
        d_o.download(out.data(), out.size());
        int bad = 0, match = 0;
        for (uint y = 0; y < h; y++)
            for (uint x = 0; x < w; x++)
                for (uint ch = 0; ch < c; ch++) {
                    float v = from_e4m3(lo[((y / 2) * (w / 2) + (x / 2)) * c + ch]) +
                              from_e4m3(skip[(y * w + x) * c + ch]);
                    u8 e = e4m3_byte(H(v));
                    if (out[(y * w + x) * c + ch] == e)
                        match++;
                    if (!std::isfinite(from_e4m3(out[(y * w + x) * c + ch])))
                        bad++;
                }
        expect(bad == 0 && match == int(w * h * c), "launch_upsample_add cpu-match");
    }

    // decoder scatter 4×4 → 8×8 C32
    {
        const uint iw = 4, ih = 4, ow = 8, oh = 8, c = 32;
        std::vector<u8> lo(iw * ih * c, q8(0.5f)), skip(ow * oh * c, q8(0.25f)), out(ow * oh * c);
        std::vector<float> sc(c, 1.f);
        DeviceBuf d_lo(lo.size()), d_sk(skip.size()), d_sc(sc.size() * 4), d_o(out.size());
        d_lo.upload(lo.data(), lo.size());
        d_sk.upload(skip.data(), skip.size());
        d_sc.upload(sc.data(), sc.size() * 4);
        launch_decoder_scatter(d_lo.as<u8>(), d_sk.as<u8>(), d_sc.as<float>(), d_o.as<u8>(), iw, ih,
                               ow, oh, c);
        hip_check(hipDeviceSynchronize(), "scatter");
        d_o.download(out.data(), out.size());
        int match = 0, bad = 0;
        for (uint i = 0; i < ow * oh * c; i++) {
            uint oc = i % c, t = i / c, ox = t % ow, oy = t / ow;
            float v = from_e4m3(lo[((oy / 2) * iw + (ox / 2)) * c + oc]);
            v = H(v + from_e4m3(skip[i]) * sc[oc]);
            u8 e = e4m3_byte(v);
            if (out[i] == e)
                match++;
            if (!std::isfinite(from_e4m3(out[i])))
                bad++;
        }
        expect(bad == 0 && match == int(ow * oh * c), "launch_decoder_scatter cpu-match");
    }

    // post70 merge
    {
        const uint n = 64, c = 32;
        std::vector<u8> main(n * c, q8(0.5f)), skip(n * c, q8(0.25f)), out(n * c);
        std::vector<float> sc(64, 1.f);
        DeviceBuf d_m(main.size()), d_s(skip.size()), d_sc(sc.size() * 4), d_o(out.size());
        d_m.upload(main.data(), main.size());
        d_s.upload(skip.data(), skip.size());
        d_sc.upload(sc.data(), sc.size() * 4);
        launch_post70_merge(d_m.as<u8>(), d_s.as<u8>(), d_sc.as<float>(), d_o.as<u8>(), n, c);
        hip_check(hipDeviceSynchronize(), "merge");
        d_o.download(out.data(), out.size());
        int match = 0;
        for (uint i = 0; i < n * c; i++) {
            uint ch = i % c;
            float v = from_e4m3(main[i]) * sc[ch] + from_e4m3(skip[i]) * sc[c + ch];
            if (out[i] == e4m3_byte(H(v)))
                match++;
        }
        expect(match == int(n * c), "launch_post70_merge cpu-match");
    }

    // linear skip
    {
        const uint tokens = 16, cin = 32, cout = 32;
        std::vector<u8> in(tokens * cin, q8(0.5f)), w(cout * cin, 0), skip(tokens * cout, q8(0.125f)),
            out(tokens * cout);
        for (uint i = 0; i < cout; i++)
            w[i * cin + i] = q8(1.f);
        DeviceBuf d_in(in.size()), d_w(w.size()), d_sk(skip.size()), d_o(out.size());
        d_in.upload(in.data(), in.size());
        d_w.upload(w.data(), w.size());
        d_sk.upload(skip.data(), skip.size());
        launch_linear_skip(d_in.as<u8>(), d_w.as<u8>(), d_sk.as<u8>(), d_o.as<u8>(), tokens, cin,
                           cout);
        hip_check(hipDeviceSynchronize(), "lin");
        d_o.download(out.data(), out.size());
        int match = 0;
        for (uint t = 0; t < tokens; t++)
            for (uint oc = 0; oc < cout; oc++) {
                float acc = from_e4m3(in[t * cin + oc]) * from_e4m3(w[oc * cin + oc]);
                acc += from_e4m3(skip[t * cout + oc]);
                if (out[t * cout + oc] == e4m3_byte(H(acc)))
                    match++;
            }
        expect(match == int(tokens * cout), "launch_linear_skip cpu-match");
    }

    {
        const uint w = 16, h = 16, c = 32;
        std::vector<u8> in(w * h * c), packed(w * h * c), back(w * h * c);
        for (size_t i = 0; i < in.size(); i++)
            in[i] = u8(i * 3);
        DeviceBuf d_in(in.size()), d_p(packed.size()), d_b(back.size());
        d_in.upload(in.data(), in.size());
        launch_pack_windows(d_in.as<u8>(), d_p.as<u8>(), w, h, c);
        launch_unpack_windows(d_p.as<u8>(), d_b.as<u8>(), w, h, c);
        hip_check(hipDeviceSynchronize(), "pack");
        d_b.download(back.data(), back.size());
        int match = 0;
        for (size_t i = 0; i < in.size(); i++)
            if (back[i] == in[i])
                match++;
        expect(match == int(in.size()), "pack/unpack roundtrip");
    }

    std::printf("%s  fails=%d\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

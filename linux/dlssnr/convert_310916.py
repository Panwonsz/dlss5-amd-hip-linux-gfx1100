#!/usr/bin/env python3
"""Weights for nvngx_dlssnr 310.9.16: the new (small) model, and the old model it also carries.

    python3 linux/dlssnr/convert_310916.py <nvngx_dlssnr.dll 310.9.16> <output dir> [--model small|old]

310.9.16 keeps its two networks as PE resources (WEIGHTS_CC_SMALL_FP8: the new default model;
WEIGHTS_HT: the 310.8.0 model, NVIDIA's preset 1) in the same record format 310.8.0 stored at a fixed
offset. The new model is built from exactly the same layer types as the old one with fewer Swin blocks
per level (2 per side instead of 4-8; ViT, pre and post unchanged), so every record is decoded by the
310.8.0 converter's own functions (convert_dll.py) after renaming each new-model block to the old block
that plays the same role:

    new  0 pre    1 C32   2 C32+ds   3 C64   4 C64+ds   5 C128   6 C128+ds   7 C256   8 C256+ds
    old  0        1       4          5       8          9        14          15       22
    new  9 C512   10 C512+head   11..18 ViT   19 dec-in   20 21 C512   22 23 C256   24 25 C128
    old  23       30             31..38       39          40 47        48 55        56 61
    new  26 27 C64   28 29 C32   30 post
    old  62 65       66 69       70

so the output directory uses the table names the HIP network already reads, plus `model-small.txt`,
which makes the network build the small topology (hip/src/network.hip). `--model old` converts the
310.9.16 copy of the old model the same way: it must reproduce the 310.8.0 tables byte for byte, which
checks the resource reader against the existing cache.

Reads the DLL's bytes only; never loads or runs it. Nothing from NVIDIA is written anywhere but the
output directory.
"""
import argparse
import hashlib
import importlib.util
import json
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('convert_dll', HERE / 'convert_dll.py')
cd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cd)

SHA_310916 = '2bc19bc09a7b89dd44f73e2ac0108fcc586e2154bdd0830f242f4f040e28789f'
LAYOUT = 'amd-consumer-derived'
SMALL_TO_OLD = {0: 0, 1: 1, 2: 4, 3: 5, 4: 8, 5: 9, 6: 14, 7: 15, 8: 22, 9: 23, 10: 30,
                **{11 + i: 31 + i for i in range(8)}, 19: 39, 20: 40, 21: 47, 22: 48, 23: 55,
                24: 56, 25: 61, 26: 62, 27: 65, 28: 66, 29: 69, 30: 70}


def pe_resources(data):
    """{resource name: bytes} for every leaf of the PE resource tree (named levels joined by '/')."""
    def u16(o): return struct.unpack_from('<H', data, o)[0]
    def u32(o): return struct.unpack_from('<I', data, o)[0]
    if data[:2] != b'MZ':
        raise RuntimeError('not a PE file')
    pe = u32(0x3c)
    if data[pe:pe + 4] != b'PE\0\0':
        raise RuntimeError('no PE signature')
    count, opt_size = u16(pe + 6), u16(pe + 20)
    opt = pe + 24
    if u16(opt) != 0x20b:
        raise RuntimeError('expected PE32+')
    rsrc_rva = u32(opt + 128)
    sections = []
    for i in range(count):
        o = opt + opt_size + 40 * i
        vs, va, size, raw = struct.unpack_from('<IIII', data, o + 8)
        sections.append((va, size, raw))
    def off(rva, size=1):
        for va, sz, raw in sections:
            if va <= rva and rva + size <= va + sz:
                return raw + rva - va
        raise RuntimeError(f'RVA {rva:#x} not in file')
    root = off(rsrc_rva)
    out = {}
    def walk(rel, path, depth):
        if depth > 6:
            raise RuntimeError('resource tree too deep')
        named, ids = struct.unpack_from('<HH', data, root + rel + 12)
        for i in range(named + ids):
            name, child = struct.unpack_from('<II', data, root + rel + 16 + 8 * i)
            if name & 0x80000000:
                o = root + (name & 0x7fffffff)
                label = data[o + 2:o + 2 + 2 * u16(o)].decode('utf-16le')
            else:
                label = str(name)
            if child & 0x80000000:
                walk(child & 0x7fffffff, path + [label], depth + 1)
            else:
                rva, size = struct.unpack_from('<II', data, root + child)
                out['/'.join(path + [label])] = data[off(rva, size):off(rva, size) + size]
    walk(0, [], 0)
    return out


def parse_records(blob):
    """The 310.8.0 record format (convert_dll.parse_archive) without its fixed 153-record set."""
    if cd._u64(blob, 0) != len(blob):
        raise RuntimeError('weights archive size mismatch')
    payloads, cursor = {}, 8
    while cursor < len(blob):
        name_len = cd._u64(blob, cursor)
        if not 1 <= name_len <= 4096:
            raise RuntimeError('invalid record name length')
        name = blob[cursor + 8:cursor + 8 + name_len].decode('ascii')
        span = cd._u64(blob, cursor + 8 + name_len)
        body = cursor + 16 + name_len
        size = cd._u64(blob, body + 8)
        if cd._u64(blob, body) != span or size % 2 or body + 20 + size > len(blob):
            raise RuntimeError(f'invalid record body: {name}')
        if cd._u32(blob, body + 16) not in (0, 1):
            raise RuntimeError(f'non-host record: {name}')
        if name in payloads:
            raise RuntimeError(f'duplicate record {name}')
        payloads[name] = blob[body + 20:body + 20 + size]
        cursor = body + span
    if cursor != len(blob):
        raise RuntimeError('archive did not end at its size')
    return payloads


def rename(payloads, mapping):
    out = {}
    for name, raw in payloads.items():
        m = re.fullmatch(r'block(\d+)\.(.*)', name)
        if not m or int(m.group(1)) not in mapping:
            raise RuntimeError(f'unexpected record {name}')
        out[f'block{mapping[int(m.group(1))]}.{m.group(2)}'] = raw
    return out


def table_ids(name):
    m = re.match(r'(?:block|post)(\d+)-', name)
    return int(m.group(1)) if m else None


def convert(payloads, cache):
    """convert_dll.convert_nvidia_dll's derived-mode table emission, over the blocks present."""
    present = {int(re.match(r'block(\d+)\.', n).group(1)) for n in payloads}
    expected = {n: c for n, c in cd.expected_tables(layout_mode=LAYOUT).items()
                if n in cd.BRIDGE_HASHES or table_ids(n) is None or table_ids(n) in present}
    entries = {}
    cache.mkdir(parents=True, exist_ok=False)
    def payload(block, layer=0):
        return payloads[f'block{block}.layer{layer}.layer']
    def emit(name, values):
        if name in entries or name not in expected or len(values) != expected[name]:
            raise RuntimeError(f'wrong table contract: {name}')
        if not all(cd.math.isfinite(v) for v in values):
            raise RuntimeError(f'nonfinite decoded coefficient: {name}')
        cd._write_f32(cache / name, values)
        entries[name] = {'count': len(values), 'sha256': cd.sha256_file(cache / name)}
    mix, pre_body = cd._preblock(payload(0))
    emit('block0-mix.audit.f32', mix)
    post_body, scales, head = cd._post70(payload(70))
    emit('post70-head.f32', head)
    emit('post70-scales.f32', scales)
    bodies = {0: pre_body, 70: post_body}
    for b, ch in [(48, 256), (56, 128), (62, 64), (66, 32)]:
        weights, body = cd._upsample(payload(b), ch)
        emit(f'block{b}-weights.f32', weights)
        bodies[b] = body
    for b, ch in cd._multi_blocks():
        if b not in present:
            continue
        raw = bodies.get(b, payload(b))
        ffn, attention = cd._unpack_multi(raw, ch, layout_mode=LAYOUT)
        emit(f'block{b}-ffn.f32', ffn)
        emit(f'block{b}-attention-components.audit.f32', cd._multi_attention_components(raw, ch))
        emit(f'block{b}-attention.f32', attention)
    for b in list(range(5)) + list(range(66, 71)):
        if b not in present:
            continue
        ffn, attention = cd._unpack_c32(bodies.get(b, payload(b)), layout_mode=LAYOUT)
        if b == 0:
            ffn[:512] = mix
        prefix = 'post70' if b == 70 else f'block{b}'
        emit(prefix + '-ffn.f32', ffn)
        emit(prefix + '-attention.f32', attention)
    emit('block4-ds.f32', cd._downsample(payload(4), 32, layout_mode=LAYOUT))
    for b, ch in [(8, 64), (14, 128), (22, 256)]:
        emit(f'block{b}-ds.f32', cd._downsample(payload(b), ch))
    for b in list(range(23, 31)) + list(range(40, 48)):
        if b not in present:
            continue
        fw, fp, aw = cd._split([payload(b, i) for i in range(4)])
        for suffix, values in [('ffwd', fw), ('ffwd-projection', fp), ('attention', aw)]:
            emit(f'block{b}-{suffix}.f32', values)
    emit('head-matrix.f32', cd._head(payload(30, 4)))
    for b in range(31, 39):
        emit(f'block{b}-expand.f32', cd._vit_expand(payload(b)))
        emit(f'block{b}-contract.f32', cd._vit_residual(payload(b, 1), 4096))
        emit(f'block{b}-qkv.f32', cd._vit_qkv(payload(b, 2)))
        emit(f'block{b}-projection.f32', cd._vit_residual(payload(b, 4), 1024))
    emit('decoder39-weights.f32', cd._decoder39(payload(39)))
    for name, values in zip(cd.BRIDGE_HASHES, cd._derived_bridge(), strict=True):
        if sys.byteorder != 'little':
            values.byteswap()
        (cache / name).write_bytes(values.tobytes())
        entries[name] = {'count': len(values), 'sha256': cd.sha256_file(cache / name)}
    missing = sorted(set(expected) - set(entries))
    if missing:
        raise RuntimeError(f'tables not produced: {missing[:8]}')
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dll', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--model', choices=['small', 'old'], default='small')
    a = ap.parse_args()
    data = a.dll.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != SHA_310916:
        sys.exit(f'not nvngx_dlssnr 310.9.16 (SHA256 {sha})')
    res = pe_resources(data)
    want = 'WEIGHTS_CC_SMALL_FP8' if a.model == 'small' else 'WEIGHTS_HT'
    hits = [k for k in res if want in k.split('/')]
    if len(hits) != 1:
        sys.exit(f'resource {want} not found; resources: {sorted(res)[:20]}')
    payloads = parse_records(res[hits[0]])
    if a.model == 'small':
        payloads = rename(payloads, SMALL_TO_OLD)
    elif len(payloads) != 153:
        sys.exit(f'old model: {len(payloads)} records, expected 153')
    if a.out.exists():
        sys.exit(f'{a.out} exists; give a new output directory')
    entries = convert(payloads, a.out)
    manifest = dict(schema=cd.DERIVED_VERSION + ('-small' if a.model == 'small' else ''), source_sha256=sha,
                    resource=want, records=len(payloads), layout_mode=LAYOUT, nvidia_equivalence=False,
                    block_map=SMALL_TO_OLD if a.model == 'small' else None, tables=entries)
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    (a.out / 'full-network.ok').write_text(cd.DERIVED_VERSION + '\n')
    if a.model == 'small':
        (a.out / 'model-small.txt').write_text('nvngx_dlssnr 310.9.16 WEIGHTS_CC_SMALL_FP8\n')
    print(f'{a.out}: {len(entries)} tables from {want} ({len(payloads)} records)')


if __name__ == '__main__':
    main()

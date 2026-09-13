"""CPU-only independent contract tests; never import upstream executables."""
import importlib.util
import math
from pathlib import Path
import struct
import unittest

SPEC = importlib.util.spec_from_file_location('convert_dll', Path(__file__).parents[1] / 'dlssnr/convert_dll.py')
assert SPEC is not None and SPEC.loader is not None
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)

class ScalarTests(unittest.TestCase):
    def test_every_binary16_encoding(self):
        for bits in range(65536):
            expected = struct.unpack('<e',struct.pack('<H',bits))[0]
            result = c._f16_to_f32(bits)
            if math.isnan(expected):
                self.assertTrue(math.isnan(result))
            else:
                self.assertEqual(result,expected)
                self.assertEqual(math.copysign(1,result),math.copysign(1,expected))
    def test_fp8_all_codes(self):
        for b in range(256):
            sign = -1 if b >= 128 else 1
            magnitude = b & 127
            value = c._e4m3(b)
            if magnitude == 127:
                self.assertTrue(math.isnan(value), hex(b))
            else:
                exponent, fraction = divmod(magnitude, 8)
                expected = sign * (fraction / 512 if exponent == 0 else (8 + fraction) * 2.0 ** (exponent - 10))
                self.assertEqual(value, expected)
    def test_half_rejects_odd_length(self):
        with self.assertRaises(ValueError):
            c._f16s(b'\x00')
    def test_scatter_rejects_truncation_duplicates(self):
        for raw, rows, cols in [(b'\x38', [0, 1], [0, 0]), (b'\x38\x38', [0, 0], [0, 0])]:
            with self.assertRaises(ValueError):
                c._scatter_e4m3(raw, 2, 1, rows, cols)

def archive_record(name, payload=b'\0\0'):
    encoded = name.encode('ascii')
    span = len(payload) + 40
    body = struct.pack('<QQI', span, len(payload), 1) + payload + struct.pack('<5I', 0, 0, 1, 0, len(payload)//2)
    return struct.pack('<Q', len(encoded)) + encoded + struct.pack('<Q', span) + body

def archive(*records):
    body = b''.join(records)
    return struct.pack('<Q', len(body)+8) + body

class ArchiveTests(unittest.TestCase):
    def test_rejects_incomplete_record_set(self):
        with self.assertRaisesRegex(RuntimeError, '153|record set'):
            c.parse_archive(archive(archive_record('block0.layer0.layer')))
    def test_rejects_duplicate_records(self):
        with self.assertRaisesRegex(RuntimeError, 'duplicate'):
            c.parse_archive(archive(*[archive_record('block0.layer0.layer')]*2))
    def test_payload_cannot_cross_body(self):
        r = bytearray(archive_record('x'))
        struct.pack_into('<Q', r, 25, 1000)
        with self.assertRaisesRegex(RuntimeError, 'payload|body'):
            c.parse_archive(archive(r))
    def test_truncated_header_and_trailing_garbage(self):
        for bad in [b'', struct.pack('<Q', 9)+b'x', archive(archive_record('x'), b'\0'*16)]:
            with self.assertRaises(RuntimeError):
                c.parse_archive(bad)

def finite_bytes(n):
    # No E4M3 NaNs; nontrivial periodic input for every packed address.
    return bytes(((i * 73 + (i >> 7) * 19) % 127) for i in range(n))

def independent_matrix(raw, rows, cols, row_bits, col_bits):
    import numpy as np
    # Gather logical coordinates from packed storage: opposite direction from converter.
    r, col = np.indices((rows, cols), dtype=np.int64)
    addresses = np.zeros((rows, cols), dtype=np.int64)
    for bit, dest in enumerate(row_bits):
        addresses |= ((r >> bit) & 1) << dest
    for bit, dest in enumerate(col_bits):
        addresses |= ((col >> bit) & 1) << dest
    values = np.frombuffer(raw, np.uint8)[addresses]
    e = ((values >> 3) & 15).astype(np.int32)
    mant = (values & 7).astype(np.float64)
    return ((1-2*(values.astype(np.int32)>>7))*np.where(e==0, mant/512, (8+mant)*np.exp2(e-10))).astype(np.float32).ravel()

class LayoutTests(unittest.TestCase):
    def helper(self, name):
        self.assertTrue(callable(getattr(c, name, None)), name+' is missing')
        return getattr(c, name)
    def test_block0_insertion_and_all_mix_coefficients(self):
        raw = bytearray(finite_bytes(21696))
        raw[8208:9232] = struct.pack('<512e', *range(512))
        mix, body = self.helper('_preblock')(bytes(raw))
        self.assertEqual(body, raw[:8208]+raw[9232:])
        expected = [0.0]*512
        for s in range(512):
            channel = (s//64)*4+(s//32%2)+(s//4%2)*2
            feature = (s//8%4)*4+s%4
            expected[channel*16+feature] = float(s)
        self.assertEqual(mix, expected)
    def test_post70_insertions_scales_and_head(self):
        raw = bytearray(21808)
        raw[:0x2050] = finite_bytes(0x2050)
        raw[0x20d0:0x5130] = finite_bytes(0x3060)
        raw[0x2050:0x20d0] = struct.pack('<64e', *range(64))
        raw[0x5130:] = struct.pack('<512e', *range(512))
        body, scales, head = self.helper('_post70')(bytes(raw))
        self.assertEqual(body, raw[:0x2050]+bytes(16)+raw[0x20d0:0x5130])
        order = [0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15,16,17,20,21,24,25,28,29,18,19,22,23,26,27,30,31]
        for i in range(64):
            self.assertEqual(scales[(i//32)*32+order[i%32]], i)
        expected=[]
        for r in (0,2,4):
            for col in range(32):
                addr=sum(((r>>b)&1)<<pos for b,pos in enumerate([2,5,6,7])) | sum(((col>>b)&1)<<pos for b,pos in enumerate([0,1,3,4,8]))
                expected.append(float(addr))
        self.assertEqual(head, expected)
    def test_upsample_internal_projection_and_body(self):
        import numpy as np
        fn=self.helper('_upsample')
        for ch,size,begin,ffskip,qkv,n in [(64,70048,0x7000,0x9000,0x70a0,61760),(128,230176,0x18000,0x20000,0x18120,197184),(256,820784,0x58000,0x78000,0x58220,689232)]:
            raw=bytearray(finite_bytes(size));raw[ffskip:ffskip+4*ch]=struct.pack('<'+str(2*ch)+'e', *range(2*ch))
            weights,body=fn(bytes(raw),ch)
            expected=bytearray(n);expected[:begin]=raw[:begin];expected[begin+16:begin+16+2*ch]=raw[ffskip:ffskip+2*ch];expected[qkv:]=raw[ffskip+4*ch:]
            self.assertEqual(body,expected)
            d=ch.bit_length()-1
            np.testing.assert_array_equal(weights[:2*ch*ch],independent_matrix(raw[begin:ffskip],ch,2*ch,[3]+list(range(6,d+5)),[1,0,4,5,2]+list(range(d+5,2*d+1))))
            for i in range(ch):
                out=(i//16)*16+(i%8)*2+(i%16//8)
                self.assertEqual(weights[2*ch*ch+out], ch+i)
    def test_upsample66_distinct_row_reordering(self):
        import numpy as np
        raw=bytearray(finite_bytes(22784));raw[0x2860:0x28a0]=struct.pack('<32e',*range(32))
        weights,body=self.helper('_upsample')(bytes(raw),32)
        self.assertEqual(body,raw[:0x2000]+raw[0x2800:0x2860]+raw[0x28a0:])
        base=independent_matrix(raw[0x2000:0x2800],32,64,[3,6,7,8,9],[1,0,4,5,2,10]).reshape(32,64)
        order=[0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15,16,17,20,21,24,25,28,29,18,19,22,23,26,27,30,31]
        matrix=np.asarray(weights[:2048]).reshape(32,64)
        for i in range(32):
            np.testing.assert_array_equal(matrix[order[i]],base[(i//16)*16+(i%8)*2+(i%16//8)])
            self.assertEqual(weights[2048+order[i]],i)
    def test_downsample_large_input_mapping(self):
        import numpy as np
        fn=self.helper('_downsample')
        for ch,offset,size in [(64,0xf130,69936),(128,0x30230,229936),(256,0xa8440,820288)]:
            raw=finite_bytes(size);d=ch.bit_length()-1
            expected=independent_matrix(raw[offset:offset+2*ch*ch],2*ch,ch,[3,6,7,8,9]+list(range(10,d+6)),[1,0,4,5,2]+list(range(d+6,2*d+1)))
            np.testing.assert_array_equal(fn(raw,ch),expected)
    def test_missing_measured_c32_maps_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError,'measured|unresolved'):
            c._unpack_c32(bytes(20672))
    def test_multi_attention_does_not_invent_skip_order(self):
        ffn, attention = c._unpack_multi(bytes(61760),64)
        self.assertIsNone(attention, 'missing measured skip map must not be guessed')
        self.assertEqual(len(ffn),9*64*64+64)

class DenseTests(unittest.TestCase):
    helper = LayoutTests.helper
    def test_vit_matrix_all_shapes(self):
        import numpy as np
        fn=self.helper('_vit_matrix')
        for inputs,outputs in [(1024,4096),(4096,1024),(1024,1024)]:
            raw=finite_bytes(inputs*outputs)
            ib,ob=inputs.bit_length()-1,outputs.bit_length()-1
            expected=independent_matrix(raw,outputs,inputs,[6,3,9,7,8]+list(range(10,ob+5)),[0,1,2,4,5]+list(range(ob+5,ib+ob)))
            np.testing.assert_array_equal(fn(raw,inputs,outputs),expected)
    def test_vit_qkv_prefix_scales_interleaving(self):
        import numpy as np
        fn=self.helper('_vit_qkv')
        raw=struct.pack('<32f',*range(32))+finite_bytes(3145728)
        result=fn(raw)
        self.assertEqual(len(result),3145728+32)
        self.assertEqual(list(result[-32:]),list(range(32)))
        chunks=np.frombuffer(raw[128:],np.uint8).reshape(-1,3,1024)
        for part in range(3):
            expected=independent_matrix(chunks[:,part].copy().tobytes(),1024,1024,[6,3,9,7,8,10,11,12,13,14],[0,1,2,4,5,15,16,17,18,19])
            np.testing.assert_array_equal(result[part*1048576:(part+1)*1048576],expected)
    def test_vit_residual_skip_gather_not_scatter(self):
        fn=self.helper('_vit_residual')
        raw=bytes(1048576)+struct.pack('<1024e',*range(1024))
        result=fn(raw,1024)
        expected=[]
        for logical in range(1024):
            raw_index=(logical&~31)|(logical&1)|((logical&2)<<2)|((logical&4)<<2)|((logical&8)>>2)|((logical&16)>>2)
            expected.append(raw_index)
        self.assertEqual(list(result[-1024:]),expected)
    def test_head_and_decoder39(self):
        import numpy as np
        raw=finite_bytes(524288)
        fn=self.helper('_head')
        np.testing.assert_array_equal(fn(raw+bytes(16)),independent_matrix(raw,1024,512,[3,6,7,8,9,10,11,12,13,14],[1,0,4,5,2,15,16,17,18]))
        fn=self.helper('_decoder39')
        result=fn(raw+struct.pack('<512e',*range(512)))
        np.testing.assert_array_equal(result[:524288],independent_matrix(raw,512,1024,[3,6,7,8,9,10,11,12,13],[1,0,4,5,2,14,15,16,17,18]))
        for i in range(512):
            self.assertEqual(result[524288+(i//16)*16+(i%8)*2+(i%16//8)],i)
    def test_dense_rejects_bad_lengths_and_expand_padding(self):
        fn=self.helper('_vit_expand')
        for raw in [bytes(4194304),bytes(4194321),bytes(4194304)+b'1'+bytes(15)]:
            with self.assertRaises(ValueError):fn(raw)
        for name,args in [('_vit_qkv',()),('_vit_residual',(1024,)),('_head',()),('_decoder39',()),('_preblock',()),('_post70',()),('_upsample',(64,)),('_downsample',(128,))]:
            with self.assertRaises(ValueError):self.helper(name)(b'',*args)

class CacheTests(unittest.TestCase):
    def test_audit_cache_cannot_be_runtime_cache(self):
        self.assertTrue(callable(getattr(c,'validate_cache',None)))
        import tempfile,json
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1]/'build/weights-audit') as folder:
            path=Path(folder)
            (path/'manifest.json').write_text(json.dumps({'schema':getattr(c,'CACHE_VERSION',None),'runtime_ready':False}))
            with self.assertRaisesRegex(RuntimeError,'incomplete|runtime|unresolved'):
                c.validate_cache(path)
    def test_manifest_covers_every_table_and_digest(self):
        self.assertTrue(callable(getattr(c,'_validate_tables',None)))
        import tempfile,hashlib
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1]/'build/weights-audit') as folder:
            path=Path(folder);data=struct.pack('<2f',1,2);(path/'x.f32').write_bytes(data)
            entry={'x.f32':{'count':2,'sha256':hashlib.sha256(data).hexdigest()}}
            c._validate_tables(path,entry,{'x.f32':2})
            for bad in [b'',data[:-1],struct.pack('<2f',1,float('nan')),struct.pack('<2f',3,4)]:
                (path/'x.f32').write_bytes(bad)
                with self.assertRaises(RuntimeError):c._validate_tables(path,entry,{'x.f32':2})
            (path/'x.f32').write_bytes(data);(path/'rogue.f32').write_bytes(data)
            with self.assertRaises(RuntimeError):c._validate_tables(path,entry,{'x.f32':2})
            with self.assertRaises(RuntimeError):c._validate_tables(path,{}, {'x.f32':2})

def amd_offset(i, o, outputs):
    # Read-only inspect_packed.py address contract, NOT converter bit maps.
    r, s = i % 32, o % 32
    return ((i//32)*(outputs//32)+o//32)*1024 + (r&3)+((r&16)>>2)+((r&12)<<2)+(s&2)*4+(s&1)*64+(s&28)*32


def amd_swap(x):
    return (x&~3) + (x%2)*2 + (x//2%2)


def amd_skip_index(channel):
    return ((((channel&~14)|((channel<<2)&8))<<1)+(channel&12))//2


def amd_fp8(byte):
    exponent, mantissa = divmod(byte & 127, 8)
    return (-1 if byte&128 else 1)*(mantissa/512 if exponent==0 else (8+mantissa)*2.0**(exponent-10))


def amd_c32_reference(raw):
    def matrix(base, rows, cols, row_map=lambda x:x, col_map=lambda x:x):
        return [amd_fp8(raw[base+amd_offset(col_map(i),row_map(o),rows)]) for o in range(rows) for i in range(cols)]
    fw = [0.0]*512 + matrix(0,128,32,amd_swap) + matrix(4096,32,128,col_map=amd_swap)
    fw += [struct.unpack_from('<e',raw,0x2010+2*amd_skip_index(c))[0] for c in range(32)]
    aw = sum((matrix(base,32,32) for base in (0x2060,0x2460,0x2860,0x4c70)), [])
    def quadrant(t):
        y,x=divmod(t,8)
        return (y//4*2+x//4)*16+y%4*4+x%4
    for q in map(quadrant,range(64)):
        for k in map(quadrant,range(64)):
            offset=q//16*2048+k//16*512+(q&7)*64+((q&8)>>1)+(k&1)*2+(k&6)*8+(k&8)
            aw.append(struct.unpack_from('<e',raw,0x2c60+offset)[0])
    aw += [struct.unpack_from('<f',raw,0x4c60)[0]]
    aw += [struct.unpack_from('<e',raw,0x5070+2*amd_skip_index(c))[0] for c in range(32)]
    return fw,aw


class ExperimentalTests(unittest.TestCase):
    def test_manifest_rejects_lost_per_table_provenance_and_scalar_records(self):
        import json,tempfile,os,copy
        tables=Path(__file__).parents[1]/'build/weights-complete/tables'
        if not (tables/'manifest.json').is_file():
            self.skipTest('requires freshly built real experimental cache')
        manifest=json.loads((tables/'manifest.json').read_text())
        with tempfile.TemporaryDirectory(dir=tables.parent) as folder:
            target=Path(folder)
            for name in manifest['tables']:
                os.link(tables/name,target/name)
            (target/'full-network.ok').write_text((tables/'full-network.ok').read_text())
            for kind in ('provenance','scalars'):
                bad=copy.deepcopy(manifest)
                if kind=='provenance':bad['tables']['block0-ffn.f32']['provenance']='upstream-captured'
                else:bad['scalar_records']={}
                (target/'manifest.json').write_text(json.dumps(bad))
                with self.subTest(kind=kind):
                    with self.assertRaises(RuntimeError):c.validate_cache(target,layout_mode='amd-consumer-derived')

    def test_experimental_cache_contract_opt_in_and_provenance(self):
        import inspect,tempfile,json
        self.assertIn('layout_mode',inspect.signature(c.convert_nvidia_dll).parameters)
        expected=c.expected_tables(layout_mode='amd-consumer-derived')
        self.assertTrue(set(c.expected_tables()).issubset(expected))
        for b in list(range(5))+list(range(66,71)):
            prefix='post70' if b==70 else f'block{b}'
            self.assertEqual(expected[prefix+'-ffn.f32'],8736)
            self.assertEqual(expected[prefix+'-attention.f32'],8225)
        self.assertEqual(expected['block4-ds.f32'],2048)
        for name in ('hwc-to-vit.i32','vit-to-hwc.i32'):self.assertEqual(expected[name],655360)
        with self.assertRaises(ValueError):c.expected_tables(layout_mode='typo')
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1]/'build/weights-complete') as folder:
            p=Path(folder);(p/'manifest.json').write_text(json.dumps({'schema':c.CACHE_VERSION,'runtime_ready':False}))
            with self.assertRaises(RuntimeError):c.validate_cache(p,layout_mode='amd-consumer-derived')
            with self.assertRaises(ValueError):c.validate_cache(p,audit_only=True,layout_mode='amd-consumer-derived')
        dll=Path('/home/guentra/nr-rocm/nvngx_dlssnr.dll')
        if dll.is_file():
            with self.assertRaisesRegex(RuntimeError,'unresolved'):c.convert_nvidia_dll(dll)
            with self.assertRaises(ValueError):c.convert_nvidia_dll(dll,audit_only=True,layout_mode='amd-consumer-derived')

    def test_bridge_exact_pinned_derived_maps_and_independent_inverse(self):
        import hashlib
        self.assertTrue(callable(getattr(c,'_derived_bridge',None)))
        forward,back=c._derived_bridge()
        self.assertEqual(len(forward),655360)
        self.assertEqual(len(back),655360)
        for index in range(655360):
            p,h=divmod(index,1024)
            t=(p&~15)|((p&8)>>3)|((p&7)<<1)
            ch=(h&~31)|((h&1)<<1)|((h&2)>>1)|((h&16)>>2)|((h&12)<<1)
            self.assertEqual(back[index],t*1024+ch)
            self.assertEqual(back[forward[index]],index)
        self.assertEqual(sum(i==v for i,v in enumerate(forward)),10240)
        for arr,digest in [(forward,'c942210afd8ffc8a2a1e4ed81df546e5e08d4c70e20f74fdddc0fe564f224ab8'),(back,'cb950400c76a6a1602ead35a817c851e552b5a301bc8e94561ed26785d1e2754')]:
            self.assertEqual(hashlib.sha256(struct.pack('<655360i',*arr)).hexdigest(),digest)

    def test_ds4_and_multi_skip_explicit_secondary_evidence(self):
        import inspect
        self.assertIn('layout_mode',inspect.signature(c._downsample).parameters)
        raw=finite_bytes(22720)
        expected=[amd_fp8(raw[0x50b0+amd_offset(i,amd_swap(o),64)]) for o in range(64) for i in range(32)]
        self.assertEqual(c._downsample(raw,32,layout_mode='amd-consumer-derived'),expected)
        for ch,size,offset in [(64,61760,0xf0b0),(128,197184,0x30130),(256,689232,0xa8240)]:
            raw=bytearray(size);raw[offset:offset+2*ch]=struct.pack('<'+str(ch)+'e',*range(ch))
            ffn,attention=c._unpack_multi(bytes(raw),ch,layout_mode='amd-consumer-derived')
            self.assertEqual(ffn,c._unpack_multi(bytes(raw),ch)[0])
            self.assertEqual(attention[:-ch],c._multi_attention_components(bytes(raw),ch))
            self.assertEqual(attention[-ch:],[amd_skip_index(amd_swap(i)) for i in range(ch)])

    def test_c32_explicit_mode_independent_physical_gather(self):
        import inspect
        self.assertIn('layout_mode',inspect.signature(c._unpack_c32).parameters)
        raw=bytearray(finite_bytes(20672))
        raw[0x2010:0x2050]=struct.pack('<32e',*range(32))
        raw[0x2c60:0x4c60]=struct.pack('<4096e',*(i%997 for i in range(4096)))
        raw[0x4c60:0x4c64]=struct.pack('<f',3.125)
        raw[0x5070:0x50b0]=struct.pack('<32e',*range(32,64))
        expected=amd_c32_reference(raw)
        self.assertEqual(c._unpack_c32(bytes(raw),layout_mode='amd-consumer-derived'),expected)
        with self.assertRaises(ValueError):c._unpack_c32(b'',layout_mode='amd-consumer-derived')
        with self.assertRaises(ValueError):c._unpack_c32(bytes(raw),layout_mode='typo')


if __name__ == '__main__':
    unittest.main()

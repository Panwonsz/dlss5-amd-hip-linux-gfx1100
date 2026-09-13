"""Real HIP enumeration, no model or game required."""
import ctypes
from pathlib import Path
import sys
import unittest

LIBRARY = Path(sys.argv.pop(1)).resolve() if len(sys.argv)>1 else Path('hip/libdlss5_hip.so').resolve()

class DeviceApiTests(unittest.TestCase):
    def test_exact_device_match_and_reject_missing(self):
        lib=ctypes.CDLL(str(LIBRARY))
        self.assertTrue(hasattr(lib,'dlss5_find_device'),'missing graphics-to-HIP adapter matching API')
        lookup=lib.dlss5_find_device
        lookup.argtypes=[ctypes.c_char_p];lookup.restype=ctypes.c_int
        lib.dlss5_last_error.restype=ctypes.c_char_p
        hip=ctypes.CDLL('/opt/rocm/lib/libamdhip64.so.7')
        count=ctypes.c_int();self.assertEqual(hip.hipGetDeviceCount(ctypes.byref(count)),0)
        names=[]
        for i in range(count.value):
            name=ctypes.create_string_buffer(256)
            self.assertEqual(hip.hipDeviceGetName(name,256,i),0)
            names.append(name.value)
        for i,name in enumerate(names):
            self.assertEqual(lookup(name),i if names.count(name)==1 else -1)
        self.assertEqual(lookup(b'nonexistent graphics adapter'),-1)
        self.assertIn(b'not found',lib.dlss5_last_error())
        self.assertEqual(lookup(None),-1)

if __name__=='__main__':unittest.main()

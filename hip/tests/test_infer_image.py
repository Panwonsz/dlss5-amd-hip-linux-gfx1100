import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'infer_image.py'


class ImageCliTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT.is_file(), 'missing runnable image inference front end')
        spec = importlib.util.spec_from_file_location('infer_image', SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_float_rgb_ppm_preserves_explicit_pixels(self):
        m = self.load_module()
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'output.ppm'
            values = np.array([[[0, .5, 1], [.25, 0, .75]]], dtype='<f4')
            m.save_image(path, values)
            self.assertEqual(path.read_bytes(), b'P6\n2 1\n255\n\x00\x80\xff\x40\x00\xbf')

    def test_invalid_output_is_never_saved(self):
        m = self.load_module()
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'output.png'
            with self.assertRaises(ValueError):
                m.save_image(path, np.array([[[np.nan, 0, 0]]], dtype='<f4'))
            self.assertFalse(path.exists())

    def test_existing_output_is_never_overwritten(self):
        m = self.load_module()
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'output.ppm'
            path.write_bytes(b'original')
            with self.assertRaises(FileExistsError):
                m.save_image(path, np.zeros((1,1,3), dtype='<f4'))
            self.assertEqual(path.read_bytes(), b'original')

    def test_bad_extension_does_not_create_an_empty_output(self):
        m = self.load_module()
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'output.invalid'
            with self.assertRaises(ValueError):
                m.save_image(path, np.zeros((1, 1, 3), dtype='<f4'))
            self.assertFalse(path.exists())

    def test_runtime_paths_support_repo_and_extracted_archive(self):
        m = self.load_module()
        self.assertTrue(hasattr(m, 'runtime_paths'), 'missing archive runtime discovery')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'linux/dlssnr').mkdir(parents=True)
            (root / 'hip').mkdir()
            library = root / 'hip/libdlss5_hip.so'
            library.touch()
            self.assertEqual(m.runtime_paths(root), (root / 'linux', library))
            archive = root / 'archive'
            (archive / 'dlssnr').mkdir(parents=True)
            (archive / 'bin').mkdir()
            bundled = archive / 'bin/libdlss5_hip.so'
            bundled.touch()
            self.assertEqual(m.runtime_paths(archive), (archive, bundled))


if __name__ == '__main__':
    unittest.main()

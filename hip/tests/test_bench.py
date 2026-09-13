"""CLI contracts are checked before touching GPU or allocating a model."""
import pathlib
import subprocess
import sys
import unittest

BINARY = pathlib.Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else pathlib.Path('hip-network70').resolve()

class BenchTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([str(BINARY), *args], capture_output=True, text=True, timeout=20)

    def test_help_documents_real_input_output(self):
        result = self.run_cli('--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--input', result.stdout)
        self.assertIn('--output', result.stdout)
        self.assertNotIn('device=', result.stdout)

    def test_bad_options_rejected(self):
        for args in [('--nonsense',), ('--weights',), ('--runs', '0'),
                     ('--seed', '-1'), ('--seed', '4294967296'),
                     ('--input', 'no-file'), ('--output', 'no-file')]:
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('device=', result.stdout)

if __name__ == '__main__':
    unittest.main()

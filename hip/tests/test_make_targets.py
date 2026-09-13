"""The verification entry points must reach every retained test family."""
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class VerificationTargets(unittest.TestCase):
    def dry_run(self, target, *variables):
        result = subprocess.run(
            ['make', '--no-print-directory', '-n', '-C', str(ROOT), target, *variables],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_check_reaches_gpu_graph_frame_api_and_cpu_tests(self):
        output = self.dry_run('check')
        for step in ('test-api', 'test-frame', 'test-graph', 'test_bench.py',
                     'test_infer_image.py', 'test_make_targets.py', '../linux/tests'):
            with self.subTest(step=step):
                self.assertIn(step, output)

    def test_model_target_builds_and_runs_real_weight_lifecycle_and_frames(self):
        output = self.dry_run('test-model', 'WEIGHTS=/tmp/model weights')
        for name in ('test_capi_lifecycle', 'test_frame_api'):
            with self.subTest(name=name):
                self.assertIn(f'tests/{name}.hip', output)
                self.assertIn(f'{name} "/tmp/model weights"', output)

    def test_model_target_requires_explicit_weights_before_build(self):
        result = subprocess.run(
            ['make', '--no-print-directory', '-C', str(ROOT), 'test-model'],
            capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('WEIGHTS=/path/to/complete/tables', result.stdout + result.stderr)
        self.assertNotIn('hipcc', result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()

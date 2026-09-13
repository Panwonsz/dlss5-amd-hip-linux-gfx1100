#!/usr/bin/env python3
import hashlib
import sys
import tempfile
import unittest
import json
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dlssnr import addon, deploy, package


def _write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _fake_package(root: Path, *, magpie=False):
    loader = 'dxgi.dll' if magpie else 'd3d12.dll'
    _write(root / 'dlss5-amd.addon64', b'MZ' + b'\0' * (64 * 1024))
    _write(root / loader, b'ReShade')
    _write(root / 'DLSS5-AMD' / 'native-game-flags.txt', b'DLSS5_TILED_WEIGHTS=1\n')
    _write(root / 'DLSS5-AMD' / 'native-game-tiled-assets' / 'block0.f16', b'\0\0')
    _write(root / 'DLSS5-AMD' / 'native-game-tiled-assets' / 'wave.cso', b'CSO')
    _write(root / 'DLSS5-D3D12-721' / 'D3D12Core.dll', b'core')
    return root


class PackageTests(unittest.TestCase):
    def test_validate_game_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _fake_package(Path(tmp) / 'pkg')
            info = package.validate(root)
            self.assertEqual(info['mode'], 'game')
            self.assertTrue(info['agility_sdk'])
            self.assertEqual(info['weights'], 1)
            self.assertEqual(info['loader'], 'd3d12.dll')

    def test_rejects_hip_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / 'amdhip64_7.dll', b'nope')
            with self.assertRaises(RuntimeError):
                package.validate(root)

    def test_magpie_needs_dxgi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _fake_package(Path(tmp) / 'pkg')
            with self.assertRaises(RuntimeError):
                package.validate(root, magpie=True)
            _fake_package(Path(tmp) / 'mag', magpie=True)
            info = package.validate(Path(tmp) / 'mag', magpie=True)
            self.assertEqual(info['loader'], 'dxgi.dll')


class DeployTests(unittest.TestCase):
    def test_wrapper_overrides_d3d12_not_hip(self):
        exe = Path('/tmp/game/Game.exe')
        text = deploy.wrapper_bytes(exe).decode()
        self.assertIn('d3d12=n,b', text)
        self.assertIn('d3d12core=n,b', text)
        self.assertNotIn('amdhip64', text)
        self.assertNotIn('LD_PRELOAD', text)
        magpie = deploy.wrapper_bytes(exe, magpie=True).decode()
        self.assertIn('dxgi=n,b', magpie)

    def test_install_and_uninstall_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pkg = _fake_package(tmp / 'pkg')
            game = tmp / 'game'
            exe = game / 'Game.exe'
            _write(exe, b'MZ')
            existing = game / 'd3d12.dll'
            _write(existing, b'original-reshade')
            result = deploy.install_package(
                exe, pkg, acknowledge_risk=True, replace_existing=True)
            self.assertTrue(result['valid'])
            self.assertTrue((game / 'dlss5-amd.addon64').is_file())
            self.assertTrue((game / 'DLSS5-AMD' / 'native-game-flags.txt').is_file())
            self.assertTrue((game / '.dlssnr-linux' / 'launch.sh').is_file())
            self.assertEqual((game / 'd3d12.dll').read_bytes(), b'ReShade')
            removed = deploy.uninstall_game(exe, yes=True)
            self.assertTrue(removed['removed'])
            self.assertEqual((game / 'd3d12.dll').read_bytes(), b'original-reshade')
            self.assertFalse((game / 'dlss5-amd.addon64').exists())
            self.assertFalse((game / '.dlssnr-linux').exists())


class BackendIntentTests(unittest.TestCase):
    def test_d3d_default_never_looks_up_hip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = _fake_package(root / 'pkg')
            exe = root / 'game' / 'Game.exe'
            _write(exe, b'MZ')
            with patch.object(deploy, 'ensure_hip_artifacts', side_effect=AssertionError('HIP lookup in D3D')):
                result = deploy.install_package(exe, pkg, acknowledge_risk=True)
            self.assertFalse(result['hip'])
            self.assertFalse(deploy.status_game(exe)['hip'])
            journal = json.loads((exe.parent / deploy.STORE / 'manifest.json').read_text())
            self.assertFalse(journal['hip'])
            self.assertFalse((exe.parent / 'dlss5_hip.dll').exists())
            self.assertFalse((exe.parent / deploy.STORE / 'lib').exists())
            self.assertNotIn('LD_PRELOAD', (exe.parent / deploy.STORE / 'launch.sh').read_text())
            self.assertTrue(any('SM 6.10' in note for note in result['notes']))
            self.assertFalse(any('HIP path:' in note for note in result['notes']))

    def test_explicit_hip_stays_hip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / 'weights'
            _write(weights / 'block0.f16', b'\0\0')
            artifacts = {key: root / 'artifacts' / key for key in ('so', 'dll', 'addon', 'reshade', 'flags')}
            for key, path in artifacts.items():
                _write(path, ('fixture-' + key).encode() if key != 'flags' else b'DLSS5_TILED_WEIGHTS=1\n')
            exe = root / 'game' / 'Game.exe'
            _write(exe, b'MZ')
            with patch.object(deploy, 'ensure_hip_artifacts', return_value=artifacts):
                result = deploy.install_package(exe, weights, acknowledge_risk=True, hip=True)
            self.assertTrue(result['hip'])
            self.assertTrue(deploy.status_game(exe)['hip'])
            self.assertEqual((exe.parent / 'dlss5_hip.dll').read_bytes(), b'fixture-dll')
            self.assertEqual((exe.parent / deploy.STORE / 'lib/libdlss5_hip.so').read_bytes(), b'fixture-so')
            wrapper = (exe.parent / deploy.STORE / 'launch.sh').read_text()
            self.assertIn('export LD_PRELOAD=', wrapper)
            self.assertIn('export DLSS5_HIP=1', wrapper)
            # A staged diagnostic must not arm known wait-based live rendering.
            for name in ('continuous-every-frame.txt', 'continuous-reset-preview.txt',
                         'temporal-history.txt', 'neural-frame-request.txt'):
                self.assertFalse((exe.parent / 'DLSS5-AMD' / name).exists(), name)
            self.assertTrue(any('unverified' in note.lower() for note in result['notes']))
            self.assertFalse((exe.parent / 'DLSS5-D3D12-721').exists())

    def test_explicit_d3d_dry_run_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = _fake_package(root / 'pkg')
            exe = root / 'game' / 'Game.exe'
            _write(exe, b'MZ')
            before = sorted(p.relative_to(root) for p in root.rglob('*'))
            result = deploy.install_package(exe, pkg, acknowledge_risk=True, hip=False, dry_run=True)
            self.assertFalse(result['hip'])
            self.assertEqual(before, sorted(p.relative_to(root) for p in root.rglob('*')))


class AddonTests(unittest.TestCase):
    def test_mingw_error_is_explicit(self):
        try:
            addon.mingw_cxx()
        except RuntimeError as exc:
            self.assertIn('mingw-w64', str(exc))
        else:
            self.assertTrue(addon.mingw_cxx())


if __name__ == '__main__':
    unittest.main()

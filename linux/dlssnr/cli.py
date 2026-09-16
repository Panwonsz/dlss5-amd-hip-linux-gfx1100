"""Linux / Proton installer for the lmxxf DLSS5-AMD ReShade add-on."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys

from . import addon, deploy, games, package

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent


def parser():
    result = argparse.ArgumentParser(
        description='Install lmxxf DLSS5-AMD (ReShade add-on) into a Proton/Wine game directory')
    commands = result.add_subparsers(dest='command')
    for name in ('list-games', 'list-protons'):
        command = commands.add_parser(name)
        command.add_argument('--steam-root', type=Path)
        command.add_argument('--json', action='store_true')
    build = commands.add_parser('build-addon', help='Cross-compile native-game.addon64 with mingw-w64')
    build.add_argument('--output', type=Path)
    build.add_argument('--json', action='store_true')
    hipb = commands.add_parser('build-hip', help='Build libdlss5_hip.so and dlss5_hip.dll')
    hipb.add_argument('--json', action='store_true')
    for name in ('install', 'doctor', 'status', 'uninstall'):
        command = commands.add_parser(name)
        command.add_argument('--appid', help='Exact Steam application ID')
        command.add_argument('--exe', type=Path, help='Game Windows x64 executable')
        command.add_argument('--game-dir', type=Path)
        command.add_argument('--steam-root', type=Path)
        command.add_argument('--json', action='store_true')
        if name == 'uninstall':
            command.add_argument('--yes', action='store_true')
        if name in ('install', 'doctor'):
            command.add_argument('--runner', '--proton', dest='proton', type=Path)
            command.add_argument('--confirm-runner', '--confirm-proton', dest='confirm_proton', action='store_true')
            command.add_argument('--package', type=Path,
                                 help='lmxxf package or a folder of network weights (.f32/.f16)')
            command.add_argument('--weights', type=Path,
                                 help='Directory of native-game-tiled-assets (HIP: ReShade is bundled)')
            command.add_argument('--allow-derived-layouts', action='store_true',
                                 help='Accept experimental reconstructed layouts, not NVIDIA equivalence')
            command.add_argument('--magpie', action='store_true',
                                 help='Magpie edition: require dxgi.dll, do not require in-game FSR')
            command.add_argument('--gpu', help='Optional GPU name for DXVK/vkd3d device filter')
            command.add_argument('--accept-risk', action='store_true')
            command.add_argument('--replace-existing', action='store_true')
            command.add_argument('--dry-run', action='store_true')
            command.add_argument('--hip', action='store_true', default=True,
                                 help='Experimental native Linux HIP (default), with the matching modified vkd3d runtime.')
            command.add_argument('--no-hip', action='store_false', dest='hip',
                                 help='Keep the D3D12 SM 6.10 path (needs the corresponding D3D12 runtime; usually fails in Proton).')
    return result


def choose(items, label, display):
    for index, item in enumerate(items, 1):
        print(f'  {index}. {display(item)}')
    answer = input(f'{label} (number; empty to cancel): ').strip()
    if not answer.isascii() or not answer.isdecimal() or not 1 <= int(answer) <= len(items):
        raise RuntimeError('Selection cancelled or invalid.')
    return items[int(answer) - 1]


def resolve_exe(args, interactive):
    if args.appid and (not args.appid.isascii() or not args.appid.isdecimal()):
        raise RuntimeError('--appid must be an exact numeric Steam ID.')
    if args.appid:
        if args.game_dir:
            raise RuntimeError('--game-dir and --appid cannot be combined.')
        entries = [g for g in games.discover_games(args.steam_root) if g['appid'] == args.appid]
        if len(entries) != 1:
            raise RuntimeError('AppID missing or ambiguous; use --steam-root or --exe without --appid.')
        return games.select_executable(entries[0]['path'], args.exe)
    if args.exe and not args.game_dir:
        path = args.exe.expanduser().absolute()
        if path.is_symlink():
            raise RuntimeError('The game executable must not be a symlink.')
        return games.select_executable(path.parent, path)
    root = args.game_dir.expanduser().absolute() if args.game_dir else PACKAGE_ROOT
    if not args.game_dir and root.name == 'linux':
        root = root.parent
    try:
        return games.select_executable(root, args.exe)
    except RuntimeError as exc:
        if not interactive or args.exe or not str(exc).startswith(('No PE x64 executable found', 'Multiple possible executables')):
            raise
        print(str(exc))
        candidates = games.find_executables(root)
        if candidates:
            selected = choose(candidates, 'Game executable', lambda p: str(p.relative_to(root)))
            return games.select_executable(root, selected)
        entered = input('Game directory or .exe path (empty to cancel): ').strip()
        if entered:
            path = Path(entered).expanduser().absolute()
            return games.select_executable(path if path.is_dir() else path.parent,
                                           None if path.is_dir() else path)
    raise RuntimeError('Specify --game-dir /path/game or --exe /path/game.exe.')


def resolve_proton(args, interactive):
    if args.proton:
        return games.validate_proton(args.proton, hip_interop=False)
    if interactive:
        print('Enter the Wine/Proton runner directory used by this game.')
        entered = input('Runner directory (or "steam"): ').strip()
        if entered and entered.casefold() != 'steam':
            return games.validate_proton(Path(entered).expanduser(), hip_interop=False)
        if not entered:
            raise RuntimeError('No runner selected; specify --runner /path/to/runner.')
    elif not args.appid and not args.steam_root:
        raise RuntimeError('Specify --runner /path/to/runner.')
    valid, errors = [], []
    for candidate in games.discover_protons(args.steam_root):
        try:
            valid.append(games.validate_proton(candidate, hip_interop=False))
        except RuntimeError as exc:
            errors.append(str(exc))
    if len(valid) == 1:
        return valid[0]
    if valid and interactive:
        return choose(valid, 'Runner', lambda p: str(p['root']))
    raise RuntimeError('Specify --runner /path/to/runner.\n' + '\n'.join(errors))


def require(accepted, interactive, question, flag):
    if accepted:
        return
    if interactive and input(question + ' [y/N] ').strip().casefold() in ('y', 'yes'):
        return
    raise RuntimeError(f'Explicit confirmation required: {flag}. {question}')


def check_host():
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('This installer requires Linux x86_64.')
    if os.geteuid() == 0:
        raise RuntimeError('Do not run this installer as sudo/root.')
    return {'system': platform.system(), 'machine': platform.machine()}


def emit(result, args):
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == 'build-addon':
        print('Add-on:', result['addon'])
        print('SHA256:', result['sha256'])
        return
    if args.command == 'build-hip':
        print('HIP library:', result.get('so'))
        print('PE trampoline:', result.get('dll'))
        return
    if args.command == 'doctor':
        print('Package:', result['package']['root'])
        print('Mode:', result['package']['mode'])
        if result['package'].get('hip'):
            print('HIP weights:', result['package'].get('weights_dir', result['package']['root']))
            print('Conversion required:', result['package'].get('conversion_required', False))
            print('D3D12 runtime: not used by HIP')
        else:
            print('Files:', result['package']['file_count'], 'weights:', result['package']['weights'])
            print('D3D12 runtime:', 'found' if result['package']['agility_sdk'] else 'missing — SM 6.10 will not initialise')
        print('Game:', result['game']['exe'])
        print('Runner:', result['proton']['root'])
        for warning in result.get('warnings', []):
            print('Warning:', warning)
        return
    if result.get('dry_run'):
        print('Dry run completed; nothing installed.')
    elif result.get('removed'):
        print('Uninstalled. Remove the wrapper from the launcher.')
    elif result.get('installed') and result.get('valid'):
        print('lmxxf package copied into the game directory.')
    else:
        print('No active managed installation.')
    if result.get('command_prefix'):
        print('Command prefix (Lutris / other, no %command%):')
        print(result['command_prefix'])
    if result.get('launch_options'):
        print('Steam launch options:')
        print(result['launch_options'])
    for warning in result.get('warnings', []):
        print('Warning:', warning)
    for note in result.get('notes', []):
        print('-', note)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments and not sys.stdin.isatty():
        print('Usage: install --exe /path/game.exe --package /path/lmxxf-release; see --help.', file=sys.stderr)
        return 2
    args = parser().parse_args(arguments or ['install'])
    interactive = sys.stdin.isatty() and not getattr(args, 'json', False)
    try:
        if args.command in ('list-games', 'list-protons'):
            if args.command == 'list-games':
                rows = games.discover_games(args.steam_root)
            else:
                rows = []
                for candidate in games.discover_protons(args.steam_root):
                    try:
                        games.validate_proton(candidate, hip_interop=False)
                        rows.append({'path': candidate, 'compatible': True})
                    except RuntimeError as exc:
                        rows.append({'path': candidate, 'compatible': False, 'reason': str(exc)})
            if args.json:
                print(json.dumps(rows, indent=2, ensure_ascii=False, default=str))
            else:
                for row in rows:
                    print(f"{row.get('appid', 'OK' if row.get('compatible') else 'INCOMPATIBLE')}  {row.get('name', '')}  {row['path']}")
                    if row.get('reason'):
                        print('  ' + row['reason'])
                if not rows:
                    print('No entries found.')
            return 0
        if args.command == 'build-addon':
            path = addon.build(args.output)
            emit({'addon': str(path), 'sha256': package.sha256(path)}, args)
            return 0
        if args.command == 'build-hip':
            import subprocess
            hip = REPO_ROOT / 'hip'
            result = subprocess.run(['make', '-C', str(hip), 'game'], check=False, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError((result.stdout + '\n' + result.stderr).strip() or 'make -C hip game failed')
            paths = deploy.hip_paths()
            emit({'so': str(paths['so']), 'dll': str(paths['dll'])}, args)
            return 0
        exe = resolve_exe(args, interactive)
        if args.command == 'status':
            emit(deploy.status_game(exe), args)
            return 0
        if args.command == 'uninstall':
            require(args.yes, interactive, 'Restore backups and remove lmxxf files?', '--yes')
            emit(deploy.uninstall_game(exe, yes=True), args)
            return 0
        host = check_host()
        evidence = games.inspect_game(exe)
        if not evidence['dx12'] and not args.magpie:
            raise RuntimeError('No static DirectX 12 evidence found; use --magpie for the window-scaler edition.')
        if not evidence['fsr_evidence'] and not args.magpie:
            raise RuntimeError('No FSR/FidelityFX evidence found. In-game hook needs FSR (or use --magpie).')
        if evidence['anti_cheat_evidence']:
            raise RuntimeError('Anti-cheat detected: refusing installation. ' + ', '.join(evidence['anti_cheat_evidence']))
        proton = resolve_proton(args, interactive)
        hip = getattr(args, 'hip', True)
        pkg = args.package or args.weights
        if pkg is None and hip:
            pkg = package.local_weights(exe)
        if pkg is None and interactive:
            entered = input('Path to nvngx_dlssnr.dll, weights folder, or lmxxf package: ').strip()
            pkg = Path(entered) if entered else None
        if pkg is None:
            raise RuntimeError('Specify --package or --weights /path/to/native-game-tiled-assets')
        warnings = []
        if hip:
            info = dict(package.inspect_weights(pkg, allow_derived_layouts=args.allow_derived_layouts),
                        mode='magpie' if args.magpie else 'game', hip=True)
            warnings.append('Bundled ReShade 6.8 add-on loader is copied as d3d12.dll (or dxgi.dll for Magpie).')
            warnings.append('Slow HIP proof of concept: optimization required; no gameplay or NVIDIA-equivalence guarantee.')
        else:
            info = package.validate(pkg, magpie=args.magpie)
            if not info['agility_sdk']:
                warnings.append('D3D12 SM 6.10 runtime missing from the package.')
        if args.command == 'doctor':
            emit({'host': host, 'game': evidence, 'proton': proton, 'package': info, 'warnings': warnings}, args)
            return 0
        if hip and args.dry_run:
            emit(dict(installed=False, dry_run=True, hip=True, package=info, warnings=warnings,
                      notes=[f"Conversion required: {info['conversion_required']}; no cache or game files written."],
                      **deploy.prefix_paths(exe)), args)
            return 0
        require(args.confirm_proton, interactive,
                f"Does your launcher use {proton['root']} for this game?", '--confirm-runner')
        require(args.accept_risk, interactive,
                'This copies ReShade + the HIP add-on next to the game. Continue?', '--accept-risk')
        if hip and info.get('conversion_required'):
            require(args.allow_derived_layouts, interactive,
                    'Accept experimental reconstructed weight layouts (not NVIDIA equivalence)?', '--allow-derived-layouts')
            args.allow_derived_layouts = True
        result = deploy.install_package(
            exe, pkg, magpie=args.magpie, replace_existing=args.replace_existing,
            acknowledge_risk=True, dry_run=args.dry_run, gpu_name=args.gpu, hip=hip,
            allow_derived_layouts=args.allow_derived_layouts)
        result['warnings'] = warnings
        result['proton'] = str(proton['root'])
        result['exe'] = str(exe)
        emit(result, args)
        return 0
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

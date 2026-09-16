"""Cross-build lmxxf's ReShade add-on on Linux (mingw-w64)."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def mingw_cxx() -> str:
    for name in ('x86_64-w64-mingw32-g++-posix', 'x86_64-w64-mingw32-g++'):
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError(
        'mingw-w64 C++ cross compiler is missing. Install g++-mingw-w64-x86-64 '
        '(posix thread model) and rerun: linux/install.sh build-addon'
    )


def build(output: Path | None = None) -> Path:
    root = repo_root()
    script = root / 'scripts' / 'build-addon-oneclick.sh'
    if not script.is_file():
        raise RuntimeError(f'Missing {script}; run this from the repository tree')
    mingw_cxx()
    output = Path(output).expanduser() if output else root / 'linux' / 'build' / 'dlss5-amd.addon64'
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ['bash', str(script), str(output)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = (result.stdout + '\n' + result.stderr).strip() or 'build-addon failed'
        raise RuntimeError(detail)
    if not output.is_file() or output.stat().st_size < 64 * 1024:
        raise RuntimeError(f'Add-on was not produced: {output}')
    return output.resolve()

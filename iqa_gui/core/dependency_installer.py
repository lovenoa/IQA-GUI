"""Install extensions with existing distribution versions constrained.

Conflicting dependencies fail resolution instead of silently upgrading the runtime.
This does not repair an already broken environment and requires restart validation.
"""
from __future__ import annotations
import importlib.metadata
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def write_constraints(path: Path, versions=None) -> Path:
    path = Path(path)
    if versions is None:
        versions = {dist.metadata['Name']: dist.version
                    for dist in importlib.metadata.distributions()
                    if dist.metadata.get('Name')}
    entries = []
    for name, version in sorted(versions.items()):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or any(c in str(version) for c in '\r\n'):
            raise ValueError('无法安全固定已安装组件版本。')
        entries.append(f'{name}=={version}')
    path.write_text('\n'.join(entries) + '\n', encoding='utf-8')
    return path


def install_packages(packages, on_progress=print) -> int:
    # Import names are not necessarily the names of their PyPI distributions.
    distribution_names = {"clip": "openai-clip"}
    packages = list(dict.fromkeys(distribution_names.get(name, name) for name in packages))
    if not packages:
        return 0
    if any(not re.fullmatch(r'[A-Za-z0-9_.-]+', name) for name in packages):
        raise ValueError('组件名无效，仅允许受支持的组件名称。')
    with tempfile.TemporaryDirectory(prefix='iqa-install-') as temp_dir:
        constraints = write_constraints(Path(temp_dir) / 'constraints.txt')
        command = [sys.executable, '-m', 'pip', 'install', '--constraint', str(constraints),
                   '--no-warn-script-location', *packages]
        on_progress('固定所有已安装组件版本；如有依赖冲突将停止，请勿强制结束安装进程。')
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', creationflags=flags)
        if process.stdout:
            for line in process.stdout:
                if line.strip():
                    on_progress(line.strip())
        return process.wait()


if __name__ == '__main__':
    raise SystemExit(install_packages(sys.argv[1:]))

#!/usr/bin/env python3
"""Run source AOT tests with pinned offline pytest, isolated from the runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

PINS = {'iniconfig-2.1.0-py3-none-any.whl': '9deba5723312380e77435581c6bf4935c94cbfab9b1ed33ef8d238ea168eb760', 'packaging-24.2-py3-none-any.whl': '09abb1bccd265c01f4a3aa3f7a7db064b36514d2cba19a2f694fe6150451a759', 'pluggy-1.5.0-py3-none-any.whl': '44e1ad92c8ca002de6377e165f3e0f1be63266ab4d554740532335b9d75ea669', 'pytest-8.3.5-py3-none-any.whl': 'c69214aa47deac29fad6c2a4f590b9c4a9fdb16a403176fe154b79c0b4d4d820'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--das35', action='store_true')
    parser.add_argument('--dependencies-only', action='store_true')
    args = parser.parse_args()
    if sys.version_info < (3, 11) or sys.platform != 'linux':
        raise SystemExit('The offline test wheel set targets the contest Linux/Python >=3.11 environment.')
    root = Path(__file__).resolve().parents[1]
    wheels = root / 'third_party/test-wheels'
    for name, digest in PINS.items():
        if hashlib.sha256((wheels / name).read_bytes()).hexdigest() != digest:
            raise SystemExit('Offline test wheel hash mismatch: ' + name)
    build = root / 'build'
    build.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='aot-test-deps-', dir=build) as tmp:
        # These pinned wheels are pure Python and need no console scripts.
        # Extract only into this throwaway prefix; restricted CPU nodes cannot
        # run pip's distribution probe because /etc/debian_version is unreadable.
        for name in PINS:
            with zipfile.ZipFile(wheels / name) as wheel:
                metadata = [n for n in wheel.namelist() if n.endswith('.dist-info/WHEEL')]
                assert len(metadata) == 1
                assert 'Root-Is-Purelib: true' in wheel.read(metadata[0]).decode()
                for item in wheel.infolist():
                    path = Path(item.filename)
                    assert not path.is_absolute() and '..' not in path.parts
                    assert not any(part.endswith('.data') for part in path.parts)
                    assert path.suffix not in ('.so', '.pyd', '.pth')
                    if item.is_dir():
                        continue
                    target = Path(tmp) / path
                    assert not target.exists(), 'Wheel path collision: ' + item.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(wheel.read(item))
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join([tmp, str(root / 'ninetoothed/src'),
                                          str(root / 'tools'), env.get('PYTHONPATH', '')])
        env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        child = r"""
import importlib, importlib.metadata, json
from pathlib import Path
import sys
prefix = Path(sys.argv[1]).resolve()
for name, expected in {'pytest':'8.3.5','iniconfig':'2.1.0','packaging':'24.2','pluggy':'1.5.0'}.items():
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(prefix), name
    assert importlib.metadata.version(name) == expected, name
print('AOT_TEST_DEPENDENCIES_PASS', json.dumps({'isolated':True,'path':str(prefix)}), flush=True)
if sys.argv[2] == '1':
    raise SystemExit(0)
if sys.argv[3] == '1':
    import torch, triton
    assert triton.__version__ == '3.5.1'
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx936')
    from ifab_triton_compat import install
    install()
import pytest
raise SystemExit(pytest.main(['-q', *sys.argv[4:]]))
"""
        result = subprocess.run([sys.executable, '-c', child, tmp,
                                 str(int(args.dependencies_only)), str(int(args.das35)),
                                 str(root / 'tests/test_aot_hip.py'),
                                 str(root / 'tests/test_h3_pointwise.py')], env=env)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

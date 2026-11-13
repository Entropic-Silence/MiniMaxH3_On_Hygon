"""Verify the complete pinned Python wheel set before offline installation."""
import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'configs/python-wheels.json').read_text())
normalize = lambda value: re.sub(r'[-_.]+', '-', value).lower()
required = {tuple(line.strip().split('==')) for line in
            (root / 'configs/bootstrap-requirements.txt').read_text().splitlines()
            if line.strip() and not line.startswith('#')}
assert {(normalize(name), version) for name, version in required} == {
    (normalize(item['name']), item['version']) for item in manifest}, 'Wheel lock differs from requirements'
for item in manifest:
    path = root / 'third_party/python-wheels' / item['file']
    assert path.is_file(), f'Missing dependency: {path}'
    assert path.stat().st_size == item['bytes'], f'Incomplete dependency: {path}'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], f'Dependency hash mismatch: {path}'
print(f'PYTHON_WHEELS_VERIFIED {len(manifest)}')

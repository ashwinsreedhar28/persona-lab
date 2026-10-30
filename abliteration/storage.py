"""Backend-independent checkpoint inventory, integrity checks and exclusive publication."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import struct
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    with Path(path).open('w') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def safe_member(root, name):
    rel = Path(name)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError(f'Unsafe checkpoint member: {name!r}')
    p = Path(root) / rel
    # HF snapshots use symlinks into the blob store; allow read-only source symlinks.
    if not p.is_file():
        raise ValueError(f'Missing checkpoint member: {name}')
    return p


def header(path):
    """Read safetensors metadata without importing a tensor backend or loading weights."""
    p = Path(path)
    with p.open('rb') as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError('Truncated safetensors header')
        n = struct.unpack('<Q', raw)[0]
        if not 2 <= n <= min(100_000_000, p.stat().st_size - 8):
            raise ValueError('Invalid safetensors header length')
        h = json.loads(f.read(n))
    size = p.stat().st_size - 8 - n
    entries = {k: v for k, v in h.items() if k != '__metadata__'}
    widths = {'BF16': 2, 'F16': 2, 'F32': 4, 'F64': 8, 'I64': 8, 'I32': 4,
              'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1, 'U32': 4, 'U64': 8, 'U16': 2}
    intervals = []
    for k, v in entries.items():
        start, end = v['data_offsets']
        count = 1
        for d in v['shape']:
            if not isinstance(d, int) or d < 0:
                raise ValueError(f'Invalid tensor shape: {k}')
            count *= d
        if v['dtype'] not in widths or end - start != count * widths[v['dtype']]:
            raise ValueError(f'Unsupported dtype or invalid byte count: {k}')
        if not 0 <= start <= end <= size:
            raise ValueError(f'Invalid tensor offsets: {k}')
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError('Noncontiguous or overlapping tensor offsets')
        cursor = end
    if cursor != size:
        raise ValueError('Unindexed safetensors payload')
    return entries


def inventory(source):
    source = Path(source)
    index = source / 'model.safetensors.index.json'
    if index.exists():
        mapping = json.loads(index.read_text())['weight_map']
    else:
        files = list(source.glob('*.safetensors'))
        if len(files) != 1:
            raise ValueError('Expected one safetensors file or an explicit shard index')
        mapping = {k: files[0].name for k in header(files[0])}
    if not mapping:
        raise ValueError('Empty checkpoint')
    result = {}
    for shard in sorted(set(mapping.values())):
        for key, spec in header(safe_member(source, shard)).items():
            if key in result or mapping.get(key) != shard:
                raise ValueError(f'Duplicate or incorrectly indexed tensor: {key}')
            result[key] = dict(spec, shard=shard)
    if set(result) != set(mapping):
        raise ValueError('Index references absent tensors')
    return result


def file_manifest(root, *, exclude=()):
    root = Path(root)
    return {str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=digest(p))
            for p in sorted(root.rglob('*')) if p.is_file() and
            str(p.relative_to(root)) not in exclude and '__pycache__' not in p.relative_to(root).parts and
            '.cache' not in p.relative_to(root).parts}


def verify_files(root, expected):
    root = Path(root)
    actual_names = set(file_manifest_names(root)) - {'manifest.json', 'COMPLETE'}
    if actual_names != set(expected):
        raise ValueError('Checkpoint file set differs from manifest')
    for name, spec in expected.items():
        p = safe_member(root, name)
        if p.is_symlink() or p.stat().st_size != spec['bytes'] or digest(p) != spec['sha256']:
            raise ValueError(f'Checkpoint integrity failure: {name}')


def file_manifest_names(root):
    return [str(p.relative_to(root)) for p in Path(root).rglob('*') if p.is_file() and
            '__pycache__' not in p.relative_to(root).parts and '.cache' not in p.relative_to(root).parts]


def destination(source, output):
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.is_symlink() or output.exists():
        raise FileExistsError(f'Output already exists: {output}; choose a new name')
    output = output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Source and output must be separate, nonoverlapping directories')
    return output


def publish(staging, output):
    """Atomic rename with OS-enforced no-replace, including an empty existing destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    src, dst = os.fsencode(staging), os.fsencode(output)
    if sys.platform == 'darwin':
        fn = libc.renamex_np
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = fn(src, dst, 4)  # RENAME_EXCL
    elif sys.platform.startswith('linux') and hasattr(libc, 'renameat2'):
        fn = libc.renameat2
        fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        result = fn(-100, src, -100, dst, 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise RuntimeError('Exclusive atomic publication requires Linux renameat2 or macOS renamex_np')
    if result:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(output))
    fd = os.open(str(Path(output).parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

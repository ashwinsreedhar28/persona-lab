"""Repository assets shared by CLI and UI; no model imports."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VECTOR_DIR=ROOT/'assets/vectors'
WEB_DIR=ROOT/'assets/web'

def resolve_vector(path):
    requested=Path(path).expanduser()
    if requested.is_file() or requested.parent!=Path('.'):
        return requested
    bundled=VECTOR_DIR/requested.name
    return bundled if bundled.is_file() else requested

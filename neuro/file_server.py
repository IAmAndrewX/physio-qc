"""
Serve NIfTI files via Streamlit's built-in static file server.

Creates symlinks in the app's ``static/nifti/`` directory with a ``.bin``
extension so Tornado does NOT transparently decompress ``.nii.gz`` files
(which would break NiiVue's parser).  Files are then available at
``/app/static/nifti/<hash>.bin``.

Requires ``[server] enableStaticServing = true`` in
``.streamlit/config.toml`` (already set).
"""

import hashlib
import os
from pathlib import Path

# static/ directory lives at the project root (next to app.py)
_STATIC_DIR = Path(__file__).resolve().parent.parent / 'static' / 'nifti'


def clear_cache():
    """Remove all .bin symlinks from the static directory."""
    if _STATIC_DIR.exists():
        for link in _STATIC_DIR.glob('*.bin'):
            try:
                link.unlink()
            except OSError:
                pass


def register_file(filepath):
    """
    Symlink a NIfTI file into the static directory and return its path.

    Parameters
    ----------
    filepath : str or Path
        Absolute path to the .nii.gz file on disk.

    Returns
    -------
    str
        Relative URL path, e.g. ``/app/static/nifti/a1b2c3d4.bin``.
        The HTML template prepends the app origin at runtime.
    """
    filepath = str(Path(filepath).resolve())
    key = hashlib.sha256(filepath.encode()).hexdigest()[:16]
    link_name = f'{key}.bin'
    link_path = _STATIC_DIR / link_name

    _STATIC_DIR.mkdir(parents=True, exist_ok=True)

    # Create symlink if it doesn't already exist
    if not link_path.exists():
        os.symlink(filepath, link_path)

    return f'/app/static/nifti/{link_name}'

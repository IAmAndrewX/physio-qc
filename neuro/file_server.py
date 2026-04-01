"""
Serve NIfTI files via Streamlit's built-in static file server.

Creates symlinks in the app's ``static/nifti/`` directory with a ``.bin``
extension so Tornado does NOT transparently decompress ``.nii.gz`` files
(which would break NiiVue's parser).  Files are then available at
``/app/static/nifti/<hash>.bin``.

NIfTI files with a missing sform (sform_code=0) are automatically repaired
by copying the qform into the sform.  The fixed file is saved alongside the
original (suffix ``_sformfix``) and served instead.

Requires ``[server] enableStaticServing = true`` in
``.streamlit/config.toml`` (already set).
"""

import hashlib
import os
from pathlib import Path

# static/ directory lives at the project root (next to app.py)
_STATIC_DIR = Path(__file__).resolve().parent.parent / 'static' / 'nifti'


def _fix_sform(filepath):
    """Return *filepath* with a valid sform, fixing it on disk if needed.

    Some coregistration pipelines write a correct qform but leave sform_code=0.
    NiiVue may misinterpret geometry when the sform is missing.  This function
    detects the condition, copies qform → sform, and saves a sibling file with
    a ``_sformfix`` suffix so the original is never modified.
    """
    import nibabel as nib

    p = Path(filepath)
    # Only check .nii / .nii.gz
    if not (p.suffix == '.nii' or p.name.endswith('.nii.gz')):
        return str(filepath)

    img = nib.load(str(filepath))
    h = img.header
    sform_code = int(h['sform_code'])
    qform_code = int(h['qform_code'])

    if sform_code != 0 or qform_code == 0:
        # sform is present, or qform is also missing — nothing to fix
        return str(filepath)

    # Build fixed filename next to original
    if p.name.endswith('.nii.gz'):
        stem = p.name[:-7]
        fixed_path = p.parent / f'{stem}_sformfix.nii.gz'
    else:
        stem = p.stem
        fixed_path = p.parent / f'{stem}_sformfix.nii'

    if fixed_path.exists():
        return str(fixed_path)

    # Copy qform → sform and save
    img.set_sform(img.get_qform(), code=qform_code)
    nib.save(img, str(fixed_path))
    return str(fixed_path)


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

    If the file has sform_code=0 (missing sform), a corrected copy is
    created first and the corrected file is served instead.

    Parameters
    ----------
    filepath : str or Path
        Absolute path to the .nii or .nii.gz file on disk.

    Returns
    -------
    str
        Relative URL path, e.g. ``/app/static/nifti/a1b2c3d4.bin``.
        The HTML template prepends the app origin at runtime.
    """
    filepath = str(Path(filepath).resolve())

    # Fix missing sform before serving
    filepath = _fix_sform(filepath)

    key = hashlib.sha256(filepath.encode()).hexdigest()[:16]
    link_name = f'{key}.bin'
    link_path = _STATIC_DIR / link_name

    _STATIC_DIR.mkdir(parents=True, exist_ok=True)

    # Create symlink if it doesn't already exist
    if not link_path.exists():
        os.symlink(filepath, link_path)

    return f'/app/static/nifti/{link_name}'

"""
Server-side NIfTI masking utility.

Applies a binary mask to a NIfTI volume so that voxels outside the mask
are set to a sentinel value (transparent in NiiVue).  Masked volumes are
cached as temporary .nii.gz files keyed by (volume_path, mask_path,
mask_opacity).
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

_CACHE_DIR = Path(tempfile.gettempdir()) / f'physio_qc_masked_{os.getuid()}'


def create_masked_volume(volume_path, mask_path, mask_opacity=1.0):
    """
    Apply a binary mask to a NIfTI volume.

    Voxels outside the mask are set to a sentinel value (-1e10) when
    mask_opacity is 1.0 (fully masked), or scaled by
    ``(1 - mask_opacity)`` for partial masking.

    Parameters
    ----------
    volume_path : str or Path
        Path to the source NIfTI volume (.nii.gz).
    mask_path : str or Path
        Path to the mask NIfTI volume (.nii.gz).
    mask_opacity : float
        Masking strength 0-1.  Default 1.0 (fully masked).

    Returns
    -------
    dict
        ``{'path': str, 'cal_min': float, 'cal_max': float}``
        where cal_min/cal_max are the robust data range of inside-mask
        voxels (2nd–98th percentile).
    """
    volume_path = str(Path(volume_path).resolve())
    mask_path = str(Path(mask_path).resolve())

    # Cache key from inputs
    key_str = f'{volume_path}|{mask_path}|{mask_opacity:.2f}'
    cache_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]
    out_path = _CACHE_DIR / f'{cache_hash}.nii.gz'
    range_path = _CACHE_DIR / f'{cache_hash}.json'

    if out_path.exists() and range_path.exists():
        with open(range_path) as f:
            data_range = json.load(f)
        return {
            'path': str(out_path),
            'cal_min': data_range['cal_min'],
            'cal_max': data_range['cal_max'],
        }

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    vol_img = nib.load(volume_path)
    mask_img = nib.load(mask_path)

    vol_data = vol_img.get_fdata(dtype=np.float32)
    mask_data = mask_img.get_fdata()

    # Binarise mask at 0.5 threshold
    binary_mask = (mask_data > 0.5).astype(np.float32)

    # Resample mask to volume space if shapes or affines differ
    if binary_mask.shape[:3] != vol_data.shape[:3] or not np.allclose(mask_img.affine, vol_img.affine):
        from nilearn.image import resample_to_img
        mask_resampled = resample_to_img(
            mask_img, vol_img, interpolation='nearest',
        )
        binary_mask = (mask_resampled.get_fdata() > 0.5).astype(np.float32)

    # Compute robust data range from inside-mask voxels
    inside = binary_mask > 0.5
    inside_vals = vol_data[inside]
    if len(inside_vals) > 0:
        data_cal_min = float(np.percentile(inside_vals, 2))
        data_cal_max = float(np.percentile(inside_vals, 98))
    else:
        data_cal_min = 0.0
        data_cal_max = 1.0

    # Apply mask: set outside voxels to a large negative sentinel value.
    # NiiVue renders below-cal_min values as transparent for overlay volumes
    # (the bottom LUT entry has alpha=0 in built-in colormaps).
    # We avoid NaN because WebGL/GPU NaN handling is inconsistent — some
    # GPUs render NaN as the maximum colormap value instead of transparent.
    _MASK_SENTINEL = np.float32(-1e10)
    outside = ~inside
    masked_data = vol_data.copy()
    if mask_opacity >= 1.0:
        masked_data[outside] = _MASK_SENTINEL
    else:
        masked_data[outside] = vol_data[outside] * (1.0 - mask_opacity)
        masked_data[outside & (masked_data == 0)] = _MASK_SENTINEL

    # Always save as float32 — NiiVue/WebGL uses float32 textures and
    # may not correctly handle NaN in float64 NIfTI files.
    masked_img = nib.Nifti1Image(masked_data.astype(np.float32), vol_img.affine)
    nib.save(masked_img, str(out_path))

    # Cache the data range alongside the volume
    with open(range_path, 'w') as f:
        json.dump({'cal_min': data_cal_min, 'cal_max': data_cal_max}, f)

    return {
        'path': str(out_path),
        'cal_min': data_cal_min,
        'cal_max': data_cal_max,
    }

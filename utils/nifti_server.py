"""
Static file helpers for the Neuro mode NiiVue viewer.

Instead of running a separate HTTP server, this module creates hard links
in Streamlit's ``static/neuro/`` directory so NIfTI files and viewer HTML
pages are served through Streamlit's built-in static file serving at
``/_app/static/neuro/...``.  This works through reverse proxies because
everything goes through the same port as Streamlit.

Hard links (not symlinks) are used because Tornado's StaticFileHandler
resolves symlinks and rejects files that resolve outside the static root.
Hard links appear as regular files and avoid this issue.
"""

import os
from pathlib import Path

# Directory inside the Streamlit project where static files are served
_STATIC_DIR = Path(__file__).resolve().parent.parent / 'static' / 'neuro'

# URL prefix for static files (Streamlit convention)
STATIC_URL_PREFIX = '/_app/static/neuro'


def prepare_static_links(image_paths, subdir=''):
    """
    Create hard links in the static directory for a set of NIfTI images.

    Clears any existing NIfTI hard links in the target subdirectory first
    to avoid stale data from a previous subject/session.

    Parameters
    ----------
    image_paths : dict
        {image_key: absolute_path_str} as returned by
        ``get_structural_images()`` or ``get_functional_images()``.
    subdir : str, optional
        Subdirectory under ``static/neuro/`` (e.g. 'struct' or 'func').
        Keeps structural and functional links separate.

    Returns
    -------
    dict
        {image_key: url_str} mapping each key to an absolute URL like
        ``/_app/static/neuro/struct/T1w.nii.gz``.
    """
    target_dir = _STATIC_DIR / subdir if subdir else _STATIC_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    # Remove old NIfTI hard links (regular .nii.gz files, not directories)
    for entry in target_dir.iterdir():
        if entry.is_file() and entry.name.endswith('.nii.gz'):
            entry.unlink()

    urls = {}
    for key, filepath in image_paths.items():
        filepath = Path(filepath)
        if not filepath.exists():
            continue
        link_name = f'{key}.nii.gz'
        link_path = target_dir / link_name
        # Remove if it somehow still exists (e.g. race condition)
        if link_path.exists():
            link_path.unlink()
        os.link(filepath, link_path)
        # Absolute URL for use in srcdoc iframes (components.html).
        if subdir:
            urls[key] = f'{STATIC_URL_PREFIX}/{subdir}/{link_name}'
        else:
            urls[key] = f'{STATIC_URL_PREFIX}/{link_name}'

    return urls


def write_viewer_page(html_content, page_name='viewer'):
    """
    Write an HTML viewer page into the static directory.

    Parameters
    ----------
    html_content : str
        Complete HTML document string.
    page_name : str
        Filename (without extension).

    Returns
    -------
    str
        URL path like ``/_app/static/neuro/viewer.html``.
    """
    _STATIC_DIR.mkdir(parents=True, exist_ok=True)
    path = _STATIC_DIR / f'{page_name}.html'
    path.write_text(html_content)
    return f'{STATIC_URL_PREFIX}/{page_name}.html'

"""
BIDS directory scanning functions for Neuro mode.
Discovers NIfTI images for structural and functional QC.
"""

import re
from pathlib import Path


def scan_bids_subjects(bids_path):
    """
    Scan the BIDS directory for available participants and sessions.

    Parameters
    ----------
    bids_path : str or Path
        Root BIDS directory (contains sub-* folders).

    Returns
    -------
    dict
        Nested dict: {participant_id: [session_ids]}
    """
    bids_path = Path(bids_path)
    if not bids_path.exists():
        return {}

    subjects = {}
    for sub_dir in sorted(bids_path.glob('sub-*')):
        if not sub_dir.is_dir():
            continue
        sessions = sorted(
            d.name for d in sub_dir.iterdir()
            if d.is_dir() and d.name.startswith('ses-')
        )
        if sessions:
            subjects[sub_dir.name] = sessions

    return subjects


def get_structural_images(bids_path, derivatives_path, participant, session,
                          smri_path=None):
    """
    Collect available structural NIfTI images for a participant/session.

    Looks in both raw BIDS and fMRIPrep derivatives. For T2starmap, SWI,
    FLAIR, and T1map, prefers coregistered ``space-T1w`` versions from
    *smri_path* (sMRI derivatives) when available, falling back to raw
    BIDS otherwise.

    Parameters
    ----------
    bids_path : str or Path
        Root BIDS directory.
    derivatives_path : str or Path
        fMRIPrep output directory (e.g. .../derivatives/fmriprep/out).
    participant : str
        Subject ID (e.g. 'sub-0011').
    session : str
        Session ID (e.g. 'ses-02').
    smri_path : str or Path or None
        sMRI derivatives directory (e.g. .../derivatives/sMRI).
        When provided, coregistered space-T1w images are preferred.

    Returns
    -------
    dict
        {image_key: absolute_path_str} for each available image.
        Keys match the overlay keys in config.STRUCTURAL_OVERLAYS plus
        'T1w' for the background.
    """
    bids_path = Path(bids_path)
    derivatives_path = Path(derivatives_path)
    images = {}

    # --- fMRIPrep anatomical ---
    # Check session-specific anat first, then session-independent (fallback)
    deriv_anat_dirs = [
        derivatives_path / participant / session / 'anat',
        derivatives_path / participant / 'anat',
    ]

    # Preprocessed T1w (background image) — prefer native T1w (no space- entity)
    for deriv_anat in deriv_anat_dirs:
        for f in sorted(deriv_anat.glob(f'{participant}*_desc-preproc_T1w.nii.gz')):
            if 'space-' not in f.name:
                images['T1w'] = str(f)
                break
        if 'T1w' in images:
            break

    # Brain mask (native space)
    for deriv_anat in deriv_anat_dirs:
        for f in sorted(deriv_anat.glob(f'{participant}*_desc-brain_mask.nii.gz')):
            if 'space-' not in f.name:
                images['brain_mask'] = str(f)
                break
        if 'brain_mask' in images:
            break

    # Tissue segmentation (discrete, native space)
    for deriv_anat in deriv_anat_dirs:
        for f in sorted(deriv_anat.glob(f'{participant}*_dseg.nii.gz')):
            if 'space-' not in f.name:
                images['dseg'] = str(f)
                break
        if 'dseg' in images:
            break

    # Probabilistic tissue maps (native space)
    for label in ('GM', 'WM', 'CSF'):
        for deriv_anat in deriv_anat_dirs:
            for f in sorted(deriv_anat.glob(f'{participant}*_label-{label}_probseg.nii.gz')):
                if 'space-' not in f.name:
                    images[f'{label}_probseg'] = str(f)
                    break
            if f'{label}_probseg' in images:
                break

    # --- Structural images: prefer coregistered space-T1w, fall back to raw ---
    smri_anat = Path(smri_path) / participant / session / 'anat' if smri_path else None
    raw_anat = bids_path / participant / session / 'anat'

    # T2* map
    found = False
    if smri_anat and smri_anat.exists():
        for f in sorted(smri_anat.glob(f'{participant}_{session}_space-T1w_T2starmap.nii.gz')):
            images['T2starmap'] = str(f)
            found = True
            break
    if not found:
        for f in sorted(raw_anat.glob(f'{participant}_{session}*_T2starmap.nii.gz')):
            images['T2starmap'] = str(f)
            break

    # SWI (ASPIRE sequence)
    found = False
    if smri_anat and smri_anat.exists():
        for f in sorted(smri_anat.glob(f'{participant}_{session}_space-T1w_acq-SWI_T2starw.nii.gz')):
            images['SWI'] = str(f)
            found = True
            break
    if not found:
        for f in sorted(raw_anat.glob(f'{participant}_{session}_acq-SWI_T2starw.nii.gz')):
            images['SWI'] = str(f)
            break

    # FLAIR
    found = False
    if smri_anat and smri_anat.exists():
        for f in sorted(smri_anat.glob(f'{participant}_{session}_space-T1w_FLAIR.nii.gz')):
            images['FLAIR'] = str(f)
            found = True
            break
    if not found:
        for f in sorted(raw_anat.glob(f'{participant}_{session}*_FLAIR.nii.gz')):
            images['FLAIR'] = str(f)
            break

    # T1map (quantitative map — no acq- entity)
    found = False
    if smri_anat and smri_anat.exists():
        for f in sorted(smri_anat.glob(f'{participant}_{session}_space-T1w_T1map.nii.gz')):
            images['T1map'] = str(f)
            found = True
            break
    if not found:
        for f in sorted(raw_anat.glob(f'{participant}_{session}*_T1map.nii.gz')):
            if 'acq-' not in f.name:
                images['T1map'] = str(f)
                break

    # QSM (Chi map from ASPIRE T2starw — in sMRI derivatives, not coregistered)
    if smri_anat and smri_anat.exists():
        for f in sorted(smri_anat.glob(f'{participant}_{session}_T2starw_Chimap.nii*')):
            images['QSM'] = str(f)
            break

    return images


def get_functional_images(derivatives_path, participant, session, task):
    """
    Collect available functional NIfTI images for a participant/session/task.

    Only returns lightweight reference images (boldref, masks), not full
    4D BOLD timeseries which are too large for browser viewing.

    Parameters
    ----------
    derivatives_path : str or Path
        fMRIPrep output directory.
    participant : str
        Subject ID (e.g. 'sub-0011').
    session : str
        Session ID (e.g. 'ses-02').
    task : str
        Task name (e.g. 'rest', 'gas', 'breath').

    Returns
    -------
    dict
        {image_key: absolute_path_str} for each available image.
    """
    derivatives_path = Path(derivatives_path)
    images = {}

    deriv_func = derivatives_path / participant / session / 'func'
    deriv_anat_dirs = [
        derivatives_path / participant / session / 'anat',
        derivatives_path / participant / 'anat',
    ]
    prefix = f'{participant}_{session}_task-{task}'

    # T1w background (from fmriprep anat — prefer native T1w)
    for deriv_anat in deriv_anat_dirs:
        for f in sorted(deriv_anat.glob(f'{participant}*_desc-preproc_T1w.nii.gz')):
            if 'space-' not in f.name:
                images['T1w'] = str(f)
                break
        if 'T1w' in images:
            break

    # T1w-space bold reference (registered to T1 — preferred for QC)
    for f in sorted(deriv_func.glob(f'{prefix}_space-T1w_boldref.nii.gz')):
        images['boldref_T1w'] = str(f)
        break

    # Brain mask
    for f in sorted(deriv_func.glob(f'{prefix}_desc-brain_mask.nii.gz')):
        images['brain_mask'] = str(f)
        break

    # T2* map from functional
    for f in sorted(deriv_func.glob(f'{prefix}_space-T1w_T2starmap.nii.gz')):
        images['T2starmap'] = str(f)
        break

    # Structural brain mask from anat (T1w space, available for masking)
    for deriv_anat in deriv_anat_dirs:
        for f in sorted(deriv_anat.glob(f'{participant}*_desc-brain_mask.nii.gz')):
            if 'space-' not in f.name:
                images['anat_brain_mask'] = str(f)
                break
        if 'anat_brain_mask' in images:
            break

    # Tissue segmentations from anat (shared across tasks)
    for deriv_anat in deriv_anat_dirs:
        for f in sorted(deriv_anat.glob(f'{participant}*_dseg.nii.gz')):
            if 'space-' not in f.name:
                images['dseg'] = str(f)
                break
        if 'dseg' in images:
            break
    for label in ('GM', 'WM', 'CSF'):
        for deriv_anat in deriv_anat_dirs:
            for f in sorted(deriv_anat.glob(f'{participant}*_label-{label}_probseg.nii.gz')):
                if 'space-' not in f.name:
                    images[f'{label}_probseg'] = str(f)
                    break
            if f'{label}_probseg' in images:
                break

    return images


def get_available_tasks(derivatives_path, participant, session):
    """
    Find which functional tasks are available for a participant/session.

    Parameters
    ----------
    derivatives_path : str or Path
        fMRIPrep output directory.
    participant : str
        Subject ID.
    session : str
        Session ID.

    Returns
    -------
    list of str
        Sorted list of task names (e.g. ['breath', 'gas', 'rest']).
    """
    deriv_func = Path(derivatives_path) / participant / session / 'func'
    if not deriv_func.exists():
        return []

    tasks = set()
    for f in deriv_func.glob('*_boldref.nii.gz'):
        match = re.search(r'task-([^_]+)', f.name)
        if match:
            tasks.add(match.group(1))

    return sorted(tasks)


def get_cvr_methods(cvr_path):
    """
    Auto-detect available CVR analysis methods (e.g. FIR, ET).

    A directory is considered a method if it contains at least one
    ``space-*`` subdirectory.

    Parameters
    ----------
    cvr_path : str or Path
        Root CVR derivatives directory.

    Returns
    -------
    list of str
        Sorted method names (e.g. ['FIR']).
    """
    cvr_path = Path(cvr_path)
    if not cvr_path.exists():
        return []

    methods = []
    for d in sorted(cvr_path.iterdir()):
        if d.is_dir() and not d.name.startswith('.'):
            if any(s.is_dir() and s.name.startswith('space-') for s in d.iterdir()):
                methods.append(d.name)
    return methods


def get_cvr_spaces(cvr_path, method):
    """
    Auto-detect available spaces for a CVR method.

    Parameters
    ----------
    cvr_path : str or Path
        Root CVR derivatives directory.
    method : str
        Method name (e.g. 'FIR').

    Returns
    -------
    list of str
        Sorted space names without the ``space-`` prefix (e.g. ['MNI']).
    """
    method_path = Path(cvr_path) / method
    if not method_path.exists():
        return []

    spaces = []
    for d in sorted(method_path.iterdir()):
        if d.is_dir() and d.name.startswith('space-'):
            spaces.append(d.name.replace('space-', ''))
    return spaces


def get_cvr_tasks(cvr_path, method, space, participant, session):
    """
    Auto-detect available tasks for a participant/session in CVR derivatives.

    Scans the session directory for ``*_stat-cvr_*`` NIfTI files and extracts
    unique ``task-<name>`` strings from filenames.

    Parameters
    ----------
    cvr_path : str or Path
        Root CVR derivatives directory.
    method : str
        CVR method (e.g. 'FIR').
    space : str
        Space name without prefix (e.g. 'MNI' or 'T1w').
    participant : str
        Subject ID (e.g. 'sub-0052').
    session : str
        Session ID (e.g. 'ses-02').

    Returns
    -------
    list of str
        Sorted task names (e.g. ['gas', 'gasTF']).
        If 'gas' is present it is listed first.
    """
    ses_dir = Path(cvr_path) / method / f'space-{space}' / participant / session
    if not ses_dir.exists():
        return []

    tasks = set()
    for f in ses_dir.glob('*_stat-cvr_*.nii.gz'):
        match = re.search(r'task-([^_]+)', f.name)
        if match:
            tasks.add(match.group(1))

    # Also check report_GLM for tasks that might only have report outputs
    report_dir = ses_dir / 'report_GLM'
    if report_dir.exists():
        for f in report_dir.glob('*_glm_r_square.nii.gz'):
            match = re.search(r'task-([^_]+)', f.name)
            if match:
                tasks.add(match.group(1))

    # Sort with 'gas' first if present
    result = sorted(tasks)
    if 'gas' in result and result[0] != 'gas':
        result.remove('gas')
        result.insert(0, 'gas')
    return result


def get_glm_images(cvr_path, fmriprep_path, participant, session, method, space,
                   task='gas'):
    """
    Collect available GLM/CVR NIfTI images for a participant/session.

    Parameters
    ----------
    cvr_path : str or Path
        Root CVR derivatives directory.
    fmriprep_path : str or Path
        fMRIPrep output directory.
    participant : str
        Subject ID (e.g. 'sub-0052').
    session : str
        Session ID (e.g. 'ses-02').
    method : str
        CVR method (e.g. 'FIR').
    space : str
        Space name without prefix (e.g. 'MNI' or 'T1w').
    task : str
        Task name (e.g. 'gas', 'gasTF'). Default 'gas'.

    Returns
    -------
    dict
        {image_key: absolute_path_str} for each available image.
    """
    cvr_path = Path(cvr_path)
    fmriprep_path = Path(fmriprep_path)
    images = {}

    ses_dir = cvr_path / method / f'space-{space}' / participant / session
    if not ses_dir.exists():
        return images

    # --- Background T1w from fmriprep ---
    # Check session-specific anat first, then session-agnostic
    anat_dirs = [
        fmriprep_path / participant / session / 'anat',
        fmriprep_path / participant / 'anat',
    ]

    if space == 'MNI':
        for anat_dir in anat_dirs:
            for f in sorted(anat_dir.glob(
                f'{participant}*_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'
            )):
                images['T1w'] = str(f)
                break
            if 'T1w' in images:
                break
    else:
        for anat_dir in anat_dirs:
            for f in sorted(anat_dir.glob(f'{participant}*_desc-preproc_T1w.nii.gz')):
                if 'space-' not in f.name:
                    images['T1w'] = str(f)
                    break
            if 'T1w' in images:
                break

    # --- CVR maps (session dir, filtered by task) ---
    task_pat = f'task-{task}_'
    for f in sorted(ses_dir.glob(
        f'{participant}*_{task_pat}*contrast-Hypercapnia_window-*_stat-cvr_*.nii.gz'
    )):
        images['cvr_hc'] = str(f)
        break

    for f in sorted(ses_dir.glob(
        f'{participant}*_{task_pat}*contrast-Hypoxia_window-*_stat-cvr_*peto2.nii.gz'
    )):
        images['cvr_hx'] = str(f)
        break

    # T-maps: beta / SE (session dir, filtered by task)
    for f in sorted(ses_dir.glob(
        f'{participant}*_{task_pat}*cnr_Hypercapnia_beta_over_se.nii.gz'
    )):
        images['tmap_hc'] = str(f)
        break

    for f in sorted(ses_dir.glob(
        f'{participant}*_{task_pat}*cnr_Hypoxia_beta_over_se.nii.gz'
    )):
        images['tmap_hx'] = str(f)
        break

    # Constant term (intercept) effect size (session dir, filtered by task)
    for f in sorted(ses_dir.glob(
        f'{participant}*_{task_pat}*contrast-constant_stat-effect_size.nii.gz'
    )):
        images['constant'] = str(f)
        break

    # F-stat maps (session dir, filtered by task)
    for f in sorted(ses_dir.glob(
        f'{participant}*_{task_pat}*contrast-Hypercapnia_window-*_stat-F.nii.gz'
    )):
        images['fstat_hc'] = str(f)
        break

    for f in sorted(ses_dir.glob(
        f'{participant}*_{task_pat}*contrast-Hypoxia_window-*_stat-F.nii.gz'
    )):
        images['fstat_hx'] = str(f)
        break

    # F z-score maps (session dir, filtered by task)
    for f in sorted(ses_dir.glob(
        f'{participant}*_{task_pat}*contrast-Hypercapnia_window-*_stat-F_zscore.nii.gz'
    )):
        images['fz_hc'] = str(f)
        break

    for f in sorted(ses_dir.glob(
        f'{participant}*_{task_pat}*contrast-Hypoxia_window-*_stat-F_zscore.nii.gz'
    )):
        images['fz_hx'] = str(f)
        break

    # Brain mask (resampled, in session dir — shared across tasks)
    for f in sorted(ses_dir.glob(f'{participant}*_desc-brain_mask_resampled.nii.gz')):
        images['brain_mask'] = str(f)
        break

    # --- Report GLM maps (filtered by task) ---
    report_dir = ses_dir / 'report_GLM'
    if report_dir.exists():
        for f in sorted(report_dir.glob(f'{participant}*_{task_pat}*cnr-Hypercapnia_over_rmsresid.nii.gz')):
            images['cnr_hc'] = str(f)
            break

        for f in sorted(report_dir.glob(f'{participant}*_{task_pat}*cnr-Hypoxia_over_rmsresid.nii.gz')):
            images['cnr_hx'] = str(f)
            break

        for f in sorted(report_dir.glob(f'{participant}*_{task_pat}*glm_r_square.nii.gz')):
            images['r_squared'] = str(f)
            break

        for f in sorted(report_dir.glob(f'{participant}*_{task_pat}*glm_residuals_rms_scaled.nii.gz')):
            images['resid_scaled'] = str(f)
            break

        for f in sorted(report_dir.glob(f'{participant}*_{task_pat}*mean_bold.nii.gz')):
            images['mean_bold'] = str(f)
            break

        # SNR: constant / RMS residuals
        for f in sorted(report_dir.glob(f'{participant}*_{task_pat}*glm_snr_constant_over_rmsresid.nii.gz')):
            images['snr'] = str(f)
            break

    return images

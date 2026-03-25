"""
Configuration file for physiological signal QC
Edit these values to match your setup and data requirements
"""

import csv as _csv
import json as _json
from pathlib import Path as _Path

# =============================================================================
# PATHS
# =============================================================================

# Path to raw physiological data files (.acq or .csv format)
# Expected structure: BASE_DATA_PATH/sub-{id}/ses-{id}/sub-{id}_ses-{id}_task-{task}_physio.acq
BASE_DATA_PATH = '/export02/projects/LCS/01_physio'

# Path where processed data will be saved (CSV + JSON)
# Output structure: OUTPUT_BASE_PATH/sub-{id}/ses-{id}/sub-{id}_ses-{id}_task-{task}_physio.{csv,json}
OUTPUT_BASE_PATH = '/export02/projects/LCS/02_physio_processed'

# Path to spirometry exports (semicolon-delimited text files, typically .csv/.xls)
SPIROMETRY_DATA_PATH = '/export02/projects/LCS/03_spirometry'

# Session labels where spirometry QC should be displayed
SPIROMETRY_SESSION_A_ALIASES = ['ses-a', 'a', 'session-a', 'ses-01', 'ses-1', 'session-1']

# Session labels that should load scanner PMU pulse/respiration into Streamlit
PMU_SESSION_B_ALIASES = ['ses-b', 'b', 'session-b', 'ses-02', 'ses-2', 'session-2']

# Siemens PMU + BIDS paths/sessions used for Session B enrichment
PMU_BIDS_BASE_PATH = '/export02/projects/LCS/BIDS'
PMU_BIDS_SESSION = 'ses-02'
PMU_PHYSIO_SESSION = 'ses-2'
PMU_SAMPLING_RATE = 400
PMU_SCAN_GAP_SECONDS = 10.0
PMU_TIME_MATCH_TOLERANCE_SECONDS = 30.0
PMU_PREFER_SCANNER_SIGNALS = True

# MRI trigger detection for sessions 02/04
# Channel matched by: name contains one of these substrings (case-insensitive)
MRI_SESSION_ALIASES = ['ses-02', 'ses-2', 'ses-04', 'ses-4']
TRIGGER_CHANNEL_PATTERNS = ['trigger', 'ami']
TRIGGER_THRESHOLD = 4.0       # Volts
TRIGGER_REFRACTORY_S = 1.5    # seconds between pulses
TRIGGER_TR = 1.72             # TR in seconds (for computing acquisition end)

# Subject-specific trigger overrides (JSON files in static/trigger_overrides/)
# Filenames: sub-{ID}.json or sub-{ID}_ses-{SS}.json
# See static/trigger_overrides/ for schema and examples.
TRIGGER_OVERRIDE_DIR = _Path(__file__).parent / 'static' / 'trigger_overrides'


def load_trigger_overrides(participant, session, task):
    """Load trigger detection overrides for a specific participant/session/task.

    Checks for (in priority order):
      1. sub-{ID}_ses-{SS}.json  (session-specific)
      2. sub-{ID}.json           (participant-wide)

    Returns dict with possible keys: threshold, refractory_s, acquisition_start,
    acquisition_end, skip_triggers.  Missing keys → use global defaults.
    """
    pid = str(participant).replace('sub-', '').strip() if participant else ''
    sess = str(session).strip().lower().replace('ses-', '').lstrip('0') or '0'
    sess_padded = sess.zfill(2)
    task_norm = str(task).strip().lower() if task else ''

    result = {}

    candidates = [
        TRIGGER_OVERRIDE_DIR / f'sub-{pid}_ses-{sess_padded}.json',
        TRIGGER_OVERRIDE_DIR / f'sub-{pid}_ses-{sess}.json',
        TRIGGER_OVERRIDE_DIR / f'sub-{pid}.json',
    ]
    data = None
    for path in candidates:
        if path.exists():
            with open(path) as f:
                data = _json.load(f)
            break
    if data is None:
        return result

    # Merge defaults then task-specific overrides
    defaults = data.get('defaults', {})
    result.update({k: v for k, v in defaults.items() if v is not None})
    task_overrides = data.get('tasks', {}).get(task_norm, {})
    result.update({k: v for k, v in task_overrides.items() if v is not None})

    return result


# Physio sessions: map to corresponding MRI session for expected duration lookup
# ses-1 → ses-02 (MRI), ses-3 → ses-04 (MRI)
PHYSIO_TO_MRI_SESSION = {
    'ses-1': '02', 'ses-01': '02', 'ses-a': '02', 'a': '02',
    'ses-3': '04', 'ses-03': '04',
}

# Fallback Biopac channel for spirometer waveform (1-based index)
SPIROMETER_CHANNEL_INDEX = 12

# Phenotype/metadata CSVs used for participant demographics and experiment notes
# Files are date-tagged (e.g. _2026-02-26_1235.csv); _latest_phenotype_file()
# resolves the newest version automatically.
PHENOTYPE_BASE_PATH = '/export02/projects/LCS/05_phenotype/redcap_exports'

def _latest_phenotype_file(prefix, ext='csv'):
    """Return the path to the newest date-tagged file matching ``prefix*.<ext>``."""
    from pathlib import Path
    matches = sorted(Path(PHENOTYPE_BASE_PATH).glob(f'{prefix}*_????-??-??_????.{ext}'))
    if matches:
        return str(matches[-1])
    # Fallback: any file starting with prefix
    fallback = sorted(Path(PHENOTYPE_BASE_PATH).glob(f'{prefix}*.{ext}'))
    return str(fallback[-1]) if fallback else f'{PHENOTYPE_BASE_PATH}/{prefix}.{ext}'

PHENOTYPE_REDCAP_PATH = _latest_phenotype_file('InvestigationOfTheLo_DATA')
PHENOTYPE_REDCAP_DEFINITIONS_PATH = _latest_phenotype_file('REDCap_variables_definitions', ext='xlsx')
PHENOTYPE_GROUP_INFO_CC_PATH = _latest_phenotype_file('Group_InfoSession_Data_CC')
PHENOTYPE_GROUP_INFO_LC_PATH = _latest_phenotype_file('Group_InfoSession_Data_LC')
PHENOTYPE_TESTING_SCHEDULE_PATH = _latest_phenotype_file('Testing_Schedule_Sheet1')
PHENOTYPE_NOTES_SESSION_A_PATH = _latest_phenotype_file('LC_Experiments_Notes_v2_Session_A_Physio')
PHENOTYPE_NOTES_SESSION_B_PATH = _latest_phenotype_file('LC_Experiments_Notes_v2_Session_B_MRI')

# Metadata view defaults
CORE_QUESTIONNAIRE_FIELDS = ['sc_tot_score', 'phq9_total_score', 'vafs']
CORE_NEUROPSYCH_FIELDS = [
    'DigitSpan_Forward',
    'DigitSpan_Backward',
    'RAVLT_DelayedRecall',
    'Category_Fluency_Category1_Total',
    'Category_Fluency_Category2_Total',
]

# ECG setup labels shown in metadata
ECG_CONFIG_CUTOFF_DATE = '2025-10-31'
ECG_CONFIG_OLD_LABEL = 'Old'
ECG_CONFIG_NEW_LABEL = 'New'

# Valsalva setup changeover (inclusive old cutoff)
VALSALVA_OLD_SETUP_CUTOFF_DATE = '2025-10-02'
VALSALVA_OLD_SETUP_NOTE = (
    'Old/wrong valsalva setup (closed-glottis test) was used through 2025-10-02.'
)
VALSALVA_NEW_SETUP_NOTE = (
    'Valsalva setup was corrected starting 2025-10-03.'
)

# =============================================================================
# DATA PARAMETERS
# =============================================================================

SAMPLING_RATE = 250

POWERLINE_FREQUENCIES = [60, 50]  # North America uses 60Hz

SIGNAL_PATTERNS = {
    'ecg': ['ecg', 'ekg', 'cardiac', 'heart'],
    'rsp': ['rsp', 'resp', 'respiratory', 'breathing', 'breath'],
    'spirometer': ['spirometer', 'spiro', 'pneumotach', 'respflow', 'maskflow', 'mask_flow'],
    'ppg': ['ppg', 'pleth', 'pulse', 'photoplethysmography'],
    'spo2': ['spo2', 'sp02', 'oxygen saturation', 'o2sat', 'saturation'],
    'bp': ['bp', 'blood_pressure', 'arterial_pressure', 'abp', 'art', 'a10' ],
    'etco2': ['co2(mmhg)', 'co2', 'etco2', 'end_tidal_co2', 'carbon_dioxide'],
    'eto2': ['o2(mmhg)', 'o2', 'eto2', 'end_tidal_o2', 'oxygen'],
    'doppler': ['doppler','a5']
}

# =============================================================================
# ECG CONFIGURATION
# =============================================================================

ECG_CLEANING_METHODS = [
    'neurokit',
    'biosppy',
    'pantompkins1985',
    'hamilton2002',
    'elgendi2010',
    'engzeemod2012',
    'vg',
    'templateconvolution',
    'custom'
]

ECG_CLEANING_INFO = {
    'neurokit': '0.5 Hz high-pass butterworth filter (order = 5), followed by powerline filtering',
    'biosppy': 'FIR filter [0.67, 45] Hz (order = 1.5 × sampling rate)',
    'pantompkins1985': 'Pan & Tompkins (1985) method',
    'hamilton2002': 'Hamilton (2002) method',
    'elgendi2010': 'Elgendi et al. (2010) method',
    'engzeemod2012': 'Engelse & Zeelenberg (1979) modified method',
    'vg': 'Visibility Graph method - 4.0 Hz high-pass butterworth (order = 2)',
    'templateconvolution': 'Template convolution method',
    'custom': 'Apply user-specified digital filters (Butterworth, FIR, Chebyshev, Elliptic, etc.)'
}

ECG_PEAK_METHODS = [
    'neurokit',
    'pantompkins1985',
    'hamilton2002',
    'zong2003',
    'martinez2004',
    'christov2004',
    'gamboa2008',
    'elgendi2010',
    'engzeemod2012',
    'manikandan2012',
    'khamis2016',
    'kalidas2017',
    'nabian2018',
    'rodrigues2021',
    'emrich2023',
    'promac'
]

ECG_PEAK_INFO = {
    'neurokit': 'NeuroKit2 default - QRS detection based on gradient steepness',
    'pantompkins1985': 'Pan & Tompkins (1985) - Classic real-time QRS detection',
    'hamilton2002': 'Hamilton (2002) algorithm',
    'zong2003': 'Zong et al. (2003) method',
    'martinez2004': 'Martinez et al. (2004) algorithm',
    'christov2004': 'Christov (2004) method',
    'gamboa2008': 'Gamboa (2008) algorithm',
    'elgendi2010': 'Elgendi et al. (2010) method',
    'engzeemod2012': 'Engelse & Zeelenberg modified by Lourenço et al. (2012)',
    'manikandan2012': 'Manikandan & Soman (2012) - Shannon energy envelope',
    'khamis2016': 'UNSW Algorithm - designed for clinical and telehealth ECGs',
    'kalidas2017': 'Kalidas et al. (2017) algorithm',
    'nabian2018': 'Nabian et al. (2018) - Pan-Tompkins adaptation',
    'rodrigues2021': 'Rodrigues et al. (2021) adaptation',
    'emrich2023': 'FastNVG - visibility graph detector (sample-accurate)',
    'promac': 'Probabilistic combination of multiple detectors'
}

DEFAULT_ECG_PARAMS = {
    'powerline': 60,
    'method': 'neurokit',
    'lowcut': 0.5,
    'highcut': 45.0,
    'peak_method': 'neurokit',
    'correct_artifacts': False,
    'calculate_quality': False,
    'rate_method': 'monotone_cubic',
    'filter_type': 'butterworth',
    'filter_order': 5,
    'apply_lowcut': True,
    'apply_highcut': True
}

# =============================================================================
# RSP CONFIGURATION
# =============================================================================

RSP_CLEANING_METHODS = ['khodadad2018', 'biosppy', 'hampel', 'custom']

RSP_CLEANING_INFO = {
    'khodadad2018': 'Second order 0.05-3 Hz bandpass Butterworth filter (NeuroKit2 default)',
    'biosppy': 'Second order 0.1-0.35 Hz bandpass Butterworth + constant detrending',
    'hampel': 'Median-based Hampel filter - replaces outliers (3 MAD from median)',
    'custom': 'Apply user-specified bandpass/lowpass/highpass filters (Butterworth, FIR, etc.)'
}

RSP_PEAK_METHODS = ['scipy', 'khodadad2018', 'biosppy']

RSP_PEAK_INFO = {
    'scipy': 'Scipy find_peaks - simple prominence-based peak detection (default)',
    'khodadad2018': 'Khodadad et al. (2018) - Optimised breath detection from impedance tomography',
    'biosppy': 'BioSPPy resp() parameters - zero-crossing based detection',
}

RSP_AMPLITUDE_METHODS = ['robust', 'standardize', 'minmax', 'none']

RSP_AMPLITUDE_INFO = {
    'robust': 'Robust normalization (median + MAD) - Best for low amplitude signals with outliers',
    'standardize': 'Z-score normalization (mean + std) - Good for consistent amplitude signals',
    'minmax': 'Min-max normalization [0, 1] - Good for very low amplitude signals',
    'none': 'No normalization - Use original signal amplitude'
}

RVT_METHODS = ['none', 'power2020', 'harrison2021', 'birn2006']

RVT_METHOD_INFO = {
    'none': 'Disable RVT computation',
    'power2020': 'Power et al. (2020) - Breath-volume normalized by breath duration, interpolated to signal length',
    'harrison2021': 'Harrison et al. (2021) - Hilbert-transform envelope approach for continuous RVT estimation',
    'birn2006': 'Birn et al. (2006) - Classic RVT: peak-to-trough amplitude divided by breath period',
}

DEFAULT_RSP_PARAMS = {
    'method': 'khodadad2018',
    'peak_method': 'scipy',
    'rate_method': 'monotone_cubic',
    'amplitude_method': 'robust',
    'lowcut': 0.05,
    'highcut': 3.0,
    'filter_type': 'butterworth',
    'filter_order': 5,
    'apply_lowcut': True,
    'apply_highcut': True,
    'rvt_method': 'none'
}

# =============================================================================
# PPG CONFIGURATION
# =============================================================================

PPG_CLEANING_METHODS = ['elgendi', 'nabian2018', 'none', 'custom']

PPG_CLEANING_INFO = {
    'elgendi': 'Elgendi et al. (2013) method (NeuroKit2 default)',
    'nabian2018': 'Nabian et al. (2018) - checks heart rate for appropriate filtering',
    'none': 'No cleaning applied - returns raw signal',
    'custom': 'Apply user-specified filters (e.g., bandpass 0.5-8 Hz) instead of NeuroKit cleaning'
}

PPG_PEAK_METHODS = ['elgendi', 'bishop', 'charlton']

PPG_PEAK_INFO = {
    'elgendi': 'Elgendi et al. (2013) systolic peak detection (default)',
    'bishop': 'Bishop & Ercole (2018) - multi-scale peak detection',
    'charlton': 'Charlton et al. (2025) MSPTDfast algorithm'
}

DEFAULT_PPG_PARAMS = {
    'method': 'elgendi',
    'peak_method': 'elgendi',
    'correct_artifacts': False,
    'rate_method': 'monotone_cubic',
    'lowcut': 0.5,
    'highcut': 8.0,
    'filter_type': 'butterworth',
    'filter_order': 5,
    'apply_lowcut': True,
    'apply_highcut': True
}

# =============================================================================
# BLOOD PRESSURE CONFIGURATION
# =============================================================================

BP_FILTER_METHODS = ['bessel_25hz', 'butterworth', 'custom']

BP_FILTER_INFO = {
    'bessel_25hz': 'Third-order Bessel lowpass at 25 Hz (used by delineator algorithm)',
    'butterworth': 'Butterworth lowpass filter (configurable cutoff frequency)',
    'custom': 'User-specified digital filters (Butterworth, FIR, Chebyshev, Elliptic, etc.)'
}

BP_PEAK_METHODS = ['delineator', 'prominence']

BP_PEAK_INFO = {
    'delineator': 'MATLAB-style delineator - derivative-based detection of systolic peaks, diastolic troughs, and dicrotic notches',
    'prominence': 'Simple prominence-based peak detection using scipy.signal.find_peaks (tunable prominence parameter)'
}

DEFAULT_BP_PARAMS = {
    'filter_method': 'bessel_25hz',
    'filter_order': 3,
    'cutoff_freq': 25,
    'peak_method': 'delineator',
    'prominence': 10,
    'detect_calibration': True,
    'calibration_threshold': 0.1,
    'calibration_min_duration': 1.0,
    'calibration_padding': 0.4,
    'noise_threshold': 0.95,
    'filter_type': 'butterworth',
    'lowcut': 0.5,
    'highcut': 15.0,
    'apply_lowcut': True,
    'apply_highcut': True
}

# =============================================================================
# GENERAL FILTER CONFIGURATION
# =============================================================================

FILTER_TYPES = ['butterworth', 'fir', 'cheby1', 'cheby2', 'elliptic', 'bessel']

DEFAULT_FILTER_PARAMS = {
    'filter_type': 'butterworth',
    'filter_order': 5,
    'apply_lowcut': True,
    'apply_highcut': True
}

# =============================================================================
# QUALITY THRESHOLDS
# =============================================================================

QUALITY_THRESHOLD_ECG = 0.5
QUALITY_THRESHOLD_BP = 0.5
QUALITY_THRESHOLD_PPG = 0.5
QUALITY_THRESHOLD_RSP = 0.5

# =============================================================================
# PEAK EDITING PARAMETERS
# =============================================================================

PEAK_ADD_WINDOW_SECONDS = 3.0
PEAK_DELETE_TOLERANCE_SECONDS = 0.5

# =============================================================================
# RATE INTERPOLATION
# =============================================================================

RATE_INTERPOLATION_METHODS = ['monotone_cubic', 'nearest', 'linear', 'quadratic', 'cubic']

RATE_INTERPOLATION_INFO = {
    'monotone_cubic': 'Monotone cubic interpolation - prevents overshoots (default, recommended)',
    'nearest': 'Nearest neighbor - step function between peaks',
    'linear': 'Linear interpolation between peaks',
    'quadratic': 'Quadratic spline interpolation',
    'cubic': 'Cubic spline interpolation'
}

# =============================================================================
# EXPORT CONFIGURATION
# =============================================================================

EXPORT_DTYPE_ONSETS = 'int8'
EXPORT_DTYPE_SIGNALS = 'float32'

PEAK_ENCODING = {
    'AUTO_DETECTED': 1,
    'MANUALLY_ADDED': 2,
    'NO_PEAK': 0,
    'DELETED': -1
}

# =============================================================================
# ETCO2 CONFIGURATION (End-Tidal CO2)
# =============================================================================

ETCO2_PEAK_METHODS = [
    'diff',        # Derivative-based with curvature filtering (recommended)
    'prominence'   # Scipy prominence-based detection
]

ETCO2_PEAK_METHOD_INFO = {
    'diff': 'Derivative zero-crossings with negative curvature filtering. Detects peaks where derivative transitions from positive to negative. More robust to baseline drift.',
    'prominence': 'Scipy prominence-based peak detection. Simpler but may be sensitive to noise and baseline variations.'
}

DEFAULT_ETCO2_PARAMS = {
    'peak_method': 'diff',
    'min_peak_distance_s': 3.0,      # Minimum 3s between breaths (20 breaths/min max)
    'min_prominence': 3.0,            # Minimum 3 mmHg prominence
    'sg_window_s': 0.2,               # 200ms Savitzky-Golay smoothing window
    'sg_poly': 2,                     # Quadratic polynomial for S-G filter
    'prom_adapt': False,              # Disable adaptive prominence by default
    'smooth_peaks': 3                 # Median filter over 3 peaks
}

# =============================================================================
# ETO2 CONFIGURATION (End-Tidal O2)
# =============================================================================

ETO2_TROUGH_METHODS = [
    'diff',        # Derivative-based with curvature filtering (recommended)
    'prominence'   # Scipy prominence-based detection on inverted signal
]

ETO2_TROUGH_METHOD_INFO = {
    'diff': 'Derivative zero-crossings with positive curvature filtering. Detects troughs (minima) where derivative transitions from negative to positive. More robust to baseline drift.',
    'prominence': 'Scipy prominence-based trough detection on inverted signal. Simpler but may be sensitive to noise.'
}

DEFAULT_ETO2_PARAMS = {
    'trough_method': 'diff',
    'min_trough_distance_s': 3.0,    # Minimum 3s between troughs (slower than peaks)
    'min_prominence': 6.0,            # Minimum 6 mmHg prominence (on inverted signal)
    'sg_window_s': 0.2,               # 200ms Savitzky-Golay smoothing window
    'sg_poly': 2,                     # Quadratic polynomial for S-G filter
    'prom_adapt': False,              # Disable adaptive prominence by default
    'smooth_troughs': 3               # Median filter over 3 troughs
}

# =============================================================================
# SPO2 CONFIGURATION (Oxygen Saturation)
# =============================================================================

SPO2_CLEANING_METHODS = ['lowpass', 'savgol', 'none']

SPO2_CLEANING_INFO = {
    'lowpass': 'Butterworth lowpass filter (removes high-frequency noise)',
    'savgol': 'Savitzky-Golay smoothing filter (preserves signal shape)',
    'none': 'No cleaning applied - use raw signal'
}

DEFAULT_SPO2_PARAMS = {
    'cleaning_method': 'lowpass',
    'lowpass_cutoff': 0.5,  # Hz - SpO2 changes slowly
    'filter_order': 2,
    'sg_window_s': 1.0,
    'sg_poly': 2,
    'desaturation_threshold': 90.0,  # % - clinical threshold
    'desaturation_drop': 3.0,  # % drop from baseline for event
    'min_event_duration_s': 10.0  # minimum duration for desaturation event
}

# =============================================================================
# TASK EVENT PROTOCOLS (onset CSV files)
# =============================================================================
# Events are loaded from BIDS-style onset CSVs: onset,duration,trial_type
# Each file lives in static/onsets/. Gas variant (short/long) is resolved
# per-participant from bids_summary.csv.

ONSET_DIR = _Path(__file__).parent / 'static' / 'onsets'

ONSET_FILES = {
    'breath':    ONSET_DIR / 'onsets_breathing.csv',
    'gas-short': ONSET_DIR / 'onsets_gas-short.csv',
    'gas-long':  ONSET_DIR / 'onsets_gas-long.csv',
    'sts':       ONSET_DIR / 'onsets_sts.csv',
    'coldpress': ONSET_DIR / 'onsets_coldpress.csv',
    'rest':      ONSET_DIR / 'onsets_rest.csv',
}

TRIAL_TYPE_COLORS = {
    'Hypercapnia': '#FF6B6B',
    'Hypoxia':     '#A78BFA',
    'Fast':        '#FF9F43',
    'Slow':        '#54A0FF',
    'Stand':       '#FF6B6B',
    'Cold':        '#60A5FA',
    'Valsalva':    '#FF6B6B',
}
TRIAL_TYPE_DEFAULT_COLOR = '#888888'

# Gas variant lookup from bids_summary.csv: nvols >= 600 → long, >= 350 → short
BIDS_SUMMARY_PATH = '/export02/projects/LCS/BIDS/code/bids_summary.csv'

def _load_gas_variant_map():
    """Build {participant_id: 'gas-long'|'gas-short'} from bids_summary.csv."""
    mapping = {}
    try:
        with open(BIDS_SUMMARY_PATH) as f:
            reader = _csv.DictReader(f)
            for row in reader:
                pid = row['ID'].strip()
                nvols = row.get('bold_task-gas_nvols', '').strip()
                if not nvols or not nvols.isdigit():
                    continue
                n = int(nvols)
                if n >= 600:
                    mapping[pid] = 'gas-long'
                elif n >= 350:
                    mapping[pid] = 'gas-short'
    except FileNotFoundError:
        pass
    return mapping

GAS_VARIANT_MAP = _load_gas_variant_map()


def _load_bids_summary():
    """Load bids_summary.csv into a list of dicts (once at import time)."""
    rows = []
    try:
        with open(BIDS_SUMMARY_PATH) as f:
            reader = _csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        pass
    return rows

_BIDS_SUMMARY_ROWS = _load_bids_summary()


def get_expected_duration(participant, session, task):
    """Look up expected scan duration (seconds) from bids_summary.csv.

    Returns float duration or None if not found.
    """
    if not _BIDS_SUMMARY_ROWS:
        return None
    pid = str(participant).replace('sub-', '').strip()
    sess_norm = str(session).strip().lower()
    # For physio sessions, map to the corresponding MRI session
    mri_sess = PHYSIO_TO_MRI_SESSION.get(sess_norm)
    if mri_sess is None:
        # Already an MRI session — extract digits
        mri_sess = sess_norm.replace('ses-', '').replace('ses', '').strip().zfill(2)
    else:
        mri_sess = mri_sess.zfill(2)
    # Normalise task key for bids_summary column lookup
    task_norm = str(task).strip().lower().replace('-', '').replace('_', '')
    # Map common aliases to bids column names
    task_col_map = {
        'gas': 'bold_task-gas_nvols', 'gaschallenge': 'bold_task-gas_nvols',
        'rest': 'bold_task-rest_nvols', 'resting': 'bold_task-rest_nvols',
        'breath': 'bold_task-breath_nvols', 'breathing': 'bold_task-breath_nvols',
    }
    col = task_col_map.get(task_norm)
    if col is None:
        return None
    for row in _BIDS_SUMMARY_ROWS:
        row_id = row.get('ID', '').strip()
        row_sess = row.get('session', '').strip().zfill(2)
        if row_id == pid and row_sess == mri_sess:
            nvols_str = row.get(col, '').strip()
            if nvols_str and nvols_str.isdigit() and int(nvols_str) > 0:
                return int(nvols_str) * TRIGGER_TR
    return None


def load_onset_events(task_key):
    """Load onset CSV → list of (onset, duration, trial_type, color) tuples."""
    path = ONSET_FILES.get(task_key)
    if not path or not _Path(path).exists():
        return []
    events = []
    with open(path) as f:
        reader = _csv.DictReader(f)
        for row in reader:
            onset = float(row['onset'])
            duration = float(row['duration'])
            trial_type = row['trial_type']
            color = TRIAL_TYPE_COLORS.get(trial_type, TRIAL_TYPE_DEFAULT_COLOR)
            events.append((onset, duration, trial_type, color))
    return events

# Map task names from filenames to protocol keys
# Normalised: lowered, stripped of hyphens/underscores
TASK_EVENT_ALIASES = {
    'breath':       'breath',
    'breathing':    'breath',
    'breathe':      'breath',
    'gas':          'gas',
    'gaschallenge': 'gas',
    'rest':         'rest',
    'resting':      'rest',
    'sts':          'sts',
    'sittostand':   'sts',
    'stand':        'sts',
    'valsalva':     'valsalva',
    'coldpress':    'coldpress',
    'coldpressor':  'coldpress',
}

# =============================================================================
# Y-AXIS LABELS (edit these to change axis titles across all plots)
# =============================================================================

Y_AXIS_LABELS = {
    # ECG
    'ecg_raw': 'Amplitude (mV)',
    'ecg_clean': 'Amplitude (mV)',
    'ecg_peaks': 'Amplitude (mV)',
    'ecg_hr': 'Heart Rate (BPM)',
    'ecg_phase': 'Cycle Completion',
    # RSP
    'rsp_raw': 'Amplitude (a.u.)',
    'rsp_clean': 'Amplitude (a.u.)',
    'rsp_peaks': 'Amplitude (a.u.)',
    'rsp_rate': 'Breathing Rate (breaths/min)',
    'rsp_rvt': 'RVT (a.u.)',
    'rsp_phase': 'Cycle Completion',
    # Spirometer
    'spiro_raw': 'Flow (a.u.)',
    'spiro_clean': 'Flow (a.u.)',
    'spiro_peaks': 'Flow (a.u.)',
    'spiro_rate': 'Breathing Rate (breaths/min)',
    'spiro_phase': 'Cycle Completion',
    # PPG
    'ppg_raw': 'Amplitude (a.u.)',
    'ppg_clean': 'Amplitude (a.u.)',
    'ppg_peaks': 'Amplitude (a.u.)',
    'ppg_hr': 'Pulse Rate (BPM)',
    # BP
    'bp_raw': 'Pressure (mmHg)',
    'bp_filtered': 'Pressure (mmHg)',
    'bp_peaks': 'Pressure (mmHg)',
    'bp_metrics': 'Pressure (mmHg)',
    'bp_hr': 'Heart Rate (BPM)',
    # Gas channels
    'etco2_raw': 'CO2 (mmHg)',
    'etco2_envelope': 'ETCO2 (mmHg)',
    'eto2_raw': 'O2 (mmHg)',
    'eto2_envelope': 'ETO2 (mmHg)',
    'spo2_raw': 'SpO2 (%)',
    'spo2_clean': 'SpO2 (%)',
}

# =============================================================================
# NEURO MODE (NIfTI Viewer)
# =============================================================================

# Path to raw BIDS dataset (contains sub-*/ses-*/anat/ and sub-*/ses-*/func/)
BIDS_DATA_PATH = '/export02/projects/LCS/BIDS'

# Path to fMRIPrep derivatives (contains sub-*/anat/ and sub-*/ses-*/func/)
FMRIPREP_DERIVATIVES_PATH = '/export02/projects/LCS/BIDS/derivatives/fmriprep/out'

# Path to sMRI derivatives (coregistered structural images in T1w space)
SMRI_DERIVATIVES_PATH = '/export02/projects/LCS/BIDS/derivatives/sMRI'

# Port for the local HTTP server that serves NIfTI files to NiiVue
NIFTI_SERVER_PORT = 8599

# Default overlay settings for the structural tab
# Entries with 'variants' show a dropdown in the gear popover to switch
# between different files for the same conceptual overlay.
STRUCTURAL_OVERLAYS = {
    'brain_mask':  {'colormap': 'red',       'opacity': 0.3, 'label': 'Brain Mask'},
    'tissue_seg':  {
        'colormap': 'freesurfer', 'opacity': 0.4, 'label': 'Tissue Segmentation',
        'variants': {
            'All (discrete)': {'key': 'dseg',        'colormap': 'freesurfer'},
            'GM Probability':  {'key': 'GM_probseg',  'colormap': 'warm'},
            'WM Probability':  {'key': 'WM_probseg',  'colormap': 'winter'},
            'CSF Probability': {'key': 'CSF_probseg', 'colormap': 'blue'},
        },
    },
    'T2starmap':   {'colormap': 'plasma',    'opacity': 0.9, 'label': 'T2* Map'},
    'SWI':         {'colormap': 'viridis',      'opacity': 0.9, 'label': 'SWI'},
    'FLAIR':       {'colormap': 'thermal',      'opacity': 0.9, 'label': 'FLAIR'},
    'T1map':       {'colormap': 'electric_blue',   'invert': True, 'opacity': 0.9, 'label': 'T1 Map', 'cal_min':750, 'cal_max': 4500},
    'QSM':         {'colormap': 'cold_hot',    'opacity': 0.9, 'label': 'QSM (Chi Map)', 'cal_min': -0.1, 'cal_max': 0.1},
}

# Default overlay settings for the functional tab
FUNCTIONAL_OVERLAYS = {
    'boldref_T1w':  {'colormap': 'inferno',   'opacity': 0.9, 'label': 'BOLD Ref (T1w space)'},
    'brain_mask':   {'colormap': 'red',    'opacity': 0.5, 'label': 'Brain Mask'},
    'tissue_seg':   {
        'colormap': 'freesurfer', 'opacity': 0.4, 'label': 'Tissue Segmentation',
        'variants': {
            'All (discrete)': {'key': 'dseg',        'colormap': 'freesurfer'},
            'GM Probability':  {'key': 'GM_probseg',  'colormap': 'warm'},
            'WM Probability':  {'key': 'WM_probseg',  'colormap': 'winter'},
            'CSF Probability': {'key': 'CSF_probseg', 'colormap': 'blue'},
        },
    },
    'T2starmap':    {'colormap': 'plasma', 'opacity': 0.9, 'label': 'T2* Map'},
}

# =============================================================================
# GLM / CVR DERIVATIVES
# =============================================================================

# Root path for CVR derivatives (contains method subdirs like FIR/, ET/)
CVR_DERIVATIVES_PATH = '/export02/projects/LCS/BIDS/derivatives/CVR'

# Default overlay settings for the GLM/CVR tab
GLM_OVERLAYS = {
    'mean_bold':    {'colormap': 'thermal',      'opacity': 0.9, 'label': 'Mean BOLD'},
    'constant':     {'colormap': 'thermal',      'opacity': 0.9, 'label': 'Constant (Intercept)'},
    'cvr_hc':       {'colormap': 'cold_hot',       'opacity': 0.9, 'label': 'CVR (Hypercapnia)',
                     'cal_min': -1.5, 'cal_max': 1.5},
    'cvr_hx':       {'colormap': 'cold_hot',       'opacity': 0.9, 'label': 'CVR (Hypoxia)',
                     'cal_min': -0.3, 'cal_max': 0.3},
    'tmap_hc':      {'colormap': 'cold_hot',       'opacity': 0.9, 'label': 'T-map (Hypercapnia)',
                     'cal_min': -5.0, 'cal_max': 5.0},
    'tmap_hx':      {'colormap': 'cold_hot',       'opacity': 0.9, 'label': 'T-map (Hypoxia)',
                     'cal_min': -5.0, 'cal_max': 5.0},
    'fstat_hc':     {'colormap': 'thermal',          'opacity': 0.9, 'label': 'F-stat (Hypercapnia)',
                     'cal_min': 0.0, 'cal_max': 100.0},
    'fstat_hx':     {'colormap': 'thermal',          'opacity': 0.9, 'label': 'F-stat (Hypoxia)',
                     'cal_min': 0.0, 'cal_max': 100.0},
    'fz_hc':        {'colormap': 'thermal',          'opacity': 0.9, 'label': 'F z-score (Hypercapnia)',
                     'cal_min': 0.0, 'cal_max': 5.0},
    'fz_hx':        {'colormap': 'thermal',          'opacity': 0.9, 'label': 'F z-score (Hypoxia)',
                     'cal_min': 0.0, 'cal_max': 5.0},
    'cnr_hc':       {'colormap': 'thermal',       'opacity': 0.9, 'label': 'CNR (Hypercapnia)',
                     'cal_min': 0, 'cal_max': 1.0},
    'cnr_hx':       {'colormap': 'thermal',       'opacity': 0.9, 'label': 'CNR (Hypoxia)',
                     'cal_min': 0, 'cal_max': 1.0},
    'snr':          {'colormap': 'thermal',    'opacity': 0.9, 'label': 'SNR (Constant/RMS Resid)'},
    'r_squared':    {'colormap': 'thermal',    'opacity': 0.9, 'label': 'R-squared', 
                     'cal_min': 0.0, 'cal_max': 1.0},
    'resid_scaled': {'colormap': 'thermal',   'opacity': 0.9, 'label': 'Scaled Residuals',
                     'cal_min': 0.0, 'cal_max': 0.1},
    'brain_mask':   {'colormap': 'red',       'opacity': 0.3, 'label': 'Brain Mask'},
}

# Available colormaps in NiiVue (subset of most useful ones)
NIIVUE_COLORMAPS = [
    'gray', 'red', 'green', 'blue', 'warm', 'electric_blue', 'cool', 'plasma',
    'viridis', 'inferno', 'winter', 'hot', 'freesurfer', 'thermal', 'blue2red',
    'cold_hot',
]

# =============================================================================
# UI THEME
# =============================================================================

THEME_COLORS = {
    'dark': {
        'background': '#0E1117',
        'secondary_bg': '#262730',
        'text': '#FAFAFA',
        'primary': '#FF4B4B'
    }
}

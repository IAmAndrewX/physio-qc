"""
Configuration file for physiological signal QC
Edit these values to match your setup and data requirements
"""

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

# Fallback Biopac channel for spirometer waveform (1-based index)
SPIROMETER_CHANNEL_INDEX = 12

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
    'min_peak_distance_s': 2.0,      # Minimum 2s between breaths (30 breaths/min max)
    'min_prominence': 1.0,            # Minimum 1 mmHg prominence
    'sg_window_s': 0.3,               # 300ms Savitzky-Golay smoothing window
    'sg_poly': 2,                     # Quadratic polynomial for S-G filter
    'prom_adapt': False,              # Disable adaptive prominence by default
    'smooth_peaks': 5                 # Median filter over 5 peaks
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
    'min_prominence': 1.0,            # Minimum 1 mmHg prominence (on inverted signal)
    'sg_window_s': 0.2,               # 200ms Savitzky-Golay smoothing window
    'sg_poly': 2,                     # Quadratic polynomial for S-G filter
    'prom_adapt': False,              # Disable adaptive prominence by default
    'smooth_troughs': 5               # Median filter over 5 troughs
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
# TASK EVENT PROTOCOLS (vertical line annotations on plots)
# =============================================================================
# Each entry: (time_in_seconds, label, hex_color)
# Timings are from protocol start (t=0 at recording start).

TASK_EVENTS = {
    # Breathing task (9 min = 540 s)
    # Sequence: [Baseline-Fast-Baseline-Slow] x2 + Final Baseline
    'breath': [
        (0,   'Baseline',       '#888888'),
        (60,  'Fast Breathing', '#FF9F43'),
        (120, 'Baseline',       '#888888'),
        (180, 'Slow Breathing', '#54A0FF'),
        (240, 'Baseline',       '#888888'),
        (300, 'Fast Breathing', '#FF9F43'),
        (360, 'Baseline',       '#888888'),
        (420, 'Slow Breathing', '#54A0FF'),
        (480, 'Baseline',       '#888888'),
    ],
    # Gas challenge (12 min = 720 s)
    # Use simple condition labels for consistency: Air / Hypercapnia / Hypoxia
    'gas': [
        (0,   'Air',         '#888888'),
        (60,  'Hypercapnia', '#FF6B6B'),
        (180, 'Air',         '#1DD1A1'),
        (300, 'Hypoxia',     '#A78BFA'),
        (480, 'Air',         '#1DD1A1'),
        (720, 'Gas Off',     '#F8F9FA'),
    ],
    # Rest (8 min = 480 s) — no mid-task events
    'rest': [],
    # STS (Session 1): 0-5 min supine, then stand at 5 min
    'sts': [
        (0,   'Supine', '#54A0FF'),
        (300, 'Stand',  '#FF6B6B'),
    ],
}

# Participant-specific task protocol overrides.
# Keys are participant prefixes (lowercase), values are {task_key: events}.
#
# sub-00 gas challenge protocol (operator table / screenshot):
# Total timeline = 1140 s with repeated Hypercapnia/Hypoxia blocks.
# Keep naming aligned to default gas labels (Air / Hypercapnia / Hypoxia).
TASK_EVENTS_PARTICIPANT_OVERRIDES = {
    'sub-00': {
        'gas': [
            (0,    'Air',         '#888888'),
            (60,   'Hypercapnia', '#FF6B6B'),
            (180,  'Air',         '#1DD1A1'),
            (240,  'Hypoxia',     '#A78BFA'),
            (420,  'Air',         '#1DD1A1'),
            (600,  'Hypercapnia', '#FF6B6B'),
            (720,  'Air',         '#1DD1A1'),
            (780,  'Hypoxia',     '#A78BFA'),
            (960,  'Air',         '#1DD1A1'),
            (1140, 'Gas Off',     '#F8F9FA'),
        ],
    },
}

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

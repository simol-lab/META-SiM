"""Analyze a single smFRET trace."""

import logging
import numpy as np
import dataclasses
import scipy.signal


MEDIAN_FILTER_WINDOW = 5  # Window size for smoothing traces with a median filter
NUM_STD_LIFETIME = 5  # Threshold for calculating lifetime
NUM_STD_MULTI_STEP = 2  # Threshold for detecting multi-step photobleaching
NUM_STD_BLINK = 1  # Threshold for detecting Cy3 blinking
MIN_FRET = 0.1  # Min FRET value to be considered valid FRET region
MIN_FRET_LENGTH = 5  # Min number of continuous frames to be considered valid FRET region
EPS = np.finfo(np.float32).eps  # Numeric tolerance for float32


@dataclasses.dataclass
class TraceStatistics:
    mean_total_intensity: float = None
    max_fret: float = None  # Highest FRET value
    first_fret: float = None  # FRET at first frame
    fret_events: int = None  # Number of FRET events
    acceptor_lifetime: float = None  # FRET acceptor lifetime
    donor_lifetime: float = None  # FRETdonor lifetime
    trace_lifetime: float = None  # FRET trace lifetime, i.e. end of the trace.
    correlation: float = None  # Correlation of Fluorophore
    correlation_derivative: float = None  # Correlation of Fluorophore derivative
    snr_background: float = None  # Signal-to-noise for the background
    snr_signal: float = None  # Signal-to-noise for the signal
    nnr_ratio: float = None  # # signal noise to background noise ratio
    background_noise: float = None  # Background noise
    cy3_blinks:  int = None  # Number of Cy3 blinks
    multi_step_photobleaching: bool = None  # Whether exists multi step photobleaching
    mean_acceptor: float = None  # Mean acceptor intensity
    mean_donor: float = None  # Mean donor intensity
    mean_fret: float = None  # Mean FRET value
    
    def asdict(self):
        """Converts the dataclass into a Python dictionary."""
        return dataclasses.asdict(self)
 
    def astuple(self):
        """Converts the dataclass into a Python tuple.
        
        The statistics names will be lost by this conversion.
        """
        return dataclasses.astuple(self)
    
    def __array__(self):
        return np.array(self.astuple())

    def __len__(self):
        return dataclasses.astuple(self).__len__()

    def __getitem__(self, item):
        return dataclasses.astuple(self).__getitem__(item)


def analyze_trace(trace) -> TraceStatistics:
    """Analyze a smFRET trace by generating the statistics of its signal.""" 
    trace_lifetime = calculate_trace_lifetime(trace)
    if trace_lifetime is None:
        return TraceStatistics()  # All the rest calculations are unsafe. Discard them now.
    multi_step_photobleaching = detect_multi_step_photobleaching(trace, trace_lifetime)
    background_noise = calculate_background_noise(trace, trace_lifetime)
    cy3_blinks = count_cy3_blinks(trace, trace_lifetime)
    donor_lifetime = calculate_donor_lifetime(trace, trace_lifetime)
    acceptor_lifetime = calculate_acceptor_lifetime(trace, trace_lifetime)
    mean_donor, mean_acceptor, mean_total_intensity = calculate_mean_intensities(trace, trace_lifetime)
    snr_background, snr_signal, nnr_ratio = calculate_snrs(trace, trace_lifetime)
    correlation, correlation_derivative = calculate_correlations(trace, trace_lifetime)
    mean_fret, max_fret, first_fret = calculate_fret_statistics(trace, trace_lifetime)
    fret_events = count_fret_events(trace, trace_lifetime)
    trace_statistics = TraceStatistics(
        trace_lifetime=trace_lifetime,
        mean_total_intensity=mean_total_intensity,
        max_fret=max_fret,
        first_fret=first_fret,
        fret_events=fret_events,
        acceptor_lifetime=acceptor_lifetime,
        donor_lifetime=donor_lifetime,
        correlation=correlation,
        correlation_derivative=correlation_derivative,
        snr_background=snr_background,
        snr_signal=snr_signal,
        nnr_ratio=nnr_ratio,
        background_noise=background_noise,
        cy3_blinks=cy3_blinks,
        multi_step_photobleaching=multi_step_photobleaching,
        mean_acceptor=mean_acceptor,
        mean_donor=mean_donor,
        mean_fret=mean_fret,
    )
    return trace_statistics
    
    
def calculate_trace_lifetime(trace):
    """Calculates donor lifetime for a smFREt trace."""
    smooth_total = scipy.signal.medfilt(trace.total, MEDIAN_FILTER_WINDOW)
    gradient_total = np.gradient(smooth_total)  # 2nd order discrete differenctiation
    mean_gradient_total = np.mean(gradient_total)
    std_gradient_total = np.std(gradient_total)
    
    non_outliers = abs(gradient_total - mean_gradient_total) <= 6.0 * std_gradient_total 
    threshold = mean_gradient_total - NUM_STD_LIFETIME * np.std(gradient_total[non_outliers])
    lifetime = np.argwhere(gradient_total <= threshold)
    if lifetime.size > 0:
        return max(2, lifetime[-1, 0]) # Found the last drop in the total intensity.
    else:
        return None

def detect_multi_step_photobleaching(trace, trace_lifetime):
    """Detect the frame of multi-step photobleaching events in a trace."""
    smooth_total = scipy.signal.medfilt(trace.total, MEDIAN_FILTER_WINDOW)
    gradient_total = np.gradient(smooth_total)  # 2nd order discrete differenctiation
    mean_gradient_total = np.mean(gradient_total)
    std_gradient_total = np.std(gradient_total)
    
    
    threshold = mean_gradient_total - NUM_STD_MULTI_STEP * std_gradient_total
    # Ignores events right at the begining or after donor lifetime.
    events = np.argwhere(gradient_total[5 : trace_lifetime] <= threshold) + 1
    
    for frame in events[::-1]:
        if frame[0] - 1 > 2:
            min_total_before = np.min(smooth_total[2 : frame[0] - 1])
        else:
            continue
        if frame[0] + 1 < trace_lifetime:
            max_total_after = np.max(smooth_total[frame[0] + 1: trace_lifetime])
        else:
            continue
        if min_total_before >= max_total_after:
            return True  # detected at least one multi-step photobleach
    return False 

def calculate_background_noise(trace, trace_lifetime):
    """Calculates the background noise of a smFRET trace."""
    if (trace_lifetime + 10) < trace.donor.size:
        start = trace_lifetime + 5  # Safely after the donor lifetime.
        noise = np.std(trace.donor[start:]) + np.std(trace.acceptor[start:])
        return noise
    else:
        return None

def count_cy3_blinks(trace, trace_lifetime: int):
    """Counts the number of Cy3 blinks in a smFRET trace."""
    if (trace_lifetime + 5) >= trace.donor.size:
        return None
    std_background = np.std(trace.total[trace_lifetime + 5:])
    blinks = trace.total[1:trace_lifetime - 1] <= NUM_STD_BLINK * std_background
    blink_starts = np.bitwise_and(blinks[:-1], ~blinks[1:])  # finds the start of each blink
    return np.sum(blink_starts)

def calculate_donor_lifetime(trace, trace_lifetime: int):
    """Calculates the length of the "donor-alive" region."""
    if (trace_lifetime + 5) >= trace.donor.size:
        return None
    std_background = np.std(trace.total[trace_lifetime + 5:])
    donor_alive = trace.total[:trace_lifetime - 1] > NUM_STD_BLINK * std_background
    return np.sum(donor_alive) 
    
def calculate_mean_intensities(trace, trace_lifetime: int):
    """Calculates the mean donor/acceptor/total intensities."""
    if (trace_lifetime + 5) >= trace.donor.size:
        return None, None, None
    std_background = np.std(trace.total[trace_lifetime + 5:])
    donor_alive = trace.total[:trace_lifetime] > NUM_STD_BLINK * std_background
    donor_alive_no_falling_edges = donor_alive[:-2] & donor_alive[1:-1] & donor_alive[2:]
    mean_donor = np.mean(trace.donor[:trace_lifetime - 2][donor_alive_no_falling_edges])
    mean_acceptor = np.mean(trace.acceptor[:trace_lifetime - 2][donor_alive_no_falling_edges])
    mean_total = mean_donor + mean_acceptor
    return (mean_donor, mean_acceptor, mean_total)

def calculate_snrs(trace, trace_lifetime: int):
    """Calculates the signal-to-noise ratios of a smFRET trace."""
    if (trace_lifetime + 5) >= trace.donor.size:
        return None, None, None
    std_background = np.std(trace.total[trace_lifetime + 5:])
    donor_alive = trace.total[:trace_lifetime] > NUM_STD_BLINK * std_background
    donor_alive_no_falling_edges = donor_alive[:-2] & donor_alive[1:-1] & donor_alive[2:]
    donor = trace.donor[:trace_lifetime - 2][donor_alive_no_falling_edges]
    acceptor = trace.acceptor[:trace_lifetime - 2][donor_alive_no_falling_edges]
    mean_total = np.mean(donor + acceptor)
    snr = mean_total / std_background  # signal to background noise
    snr_signal = mean_total / np.std(donor + acceptor)  # signal to signal noise
    nnr = np.std(donor + acceptor) / std_background  # signal noise to background noise ratio
    return (snr, snr_signal, nnr)
    
def calculate_correlations(trace, trace_lifetime: int):
    """Calculates the correlations of donor v.s. acceptor, and between their derivatives."""
    if (trace_lifetime + 5) >= trace.donor.size:
        return None, None
    std_background = np.std(trace.total[trace_lifetime + 5:])
    donor_alive = trace.total[:trace_lifetime] > NUM_STD_BLINK * std_background
    donor_alive_no_falling_edges = donor_alive[:-2] & donor_alive[1:-1] & donor_alive[2:]
    if np.sum(donor_alive_no_falling_edges) < MIN_FRET_LENGTH:
        return (None, None)  # Length of donor-alive region too small to be meaningful
    donor = trace.donor[:trace_lifetime - 2][donor_alive_no_falling_edges]
    acceptor = trace.acceptor[:trace_lifetime - 2][donor_alive_no_falling_edges]
    gradient_donor = np.gradient(donor)
    gradient_acceptor = np.gradient(acceptor)
    correlation = np.corrcoef(donor, acceptor)[0, 1]
    gradient_correlation = np.corrcoef(gradient_donor, gradient_acceptor)[0, 1]
    return (correlation, gradient_correlation)


def run_length_encode(v: np.ndarray):
    """Generates the run-length encoding for a 1-D numpy array."""
    n = v.size
    encoding = np.zeros(v.size)
    running_sum = 0
    for i in range(n)[::-1]:
        if v[i] == 0:
            running_sum = 0
        else:
            running_sum += 1
            encoding[i] = running_sum
    return encoding

def run_length_filter(v: np.ndarray, min_length: int):
    """Filters a 1-D numpy array by minimum run length of a continuous region."""
    encoding = run_length_encode(v)
    filtered_v = np.zeros_like(v)
    running_max = 0
    for i in range(v.size):
        if v[i] == 0:
            running_max = 0
        else:
            running_max = max(running_max, encoding[i])
            filtered_v[i] = (running_max >= min_length)
    return filtered_v
 
def calculate_fret_statistics(trace, trace_lifetime: int):
    """Calculates the average/max/first-frame FRET value."""
    valid_fret_bool = trace.fret[:trace_lifetime] >= MIN_FRET
    valid_fret_bool = run_length_filter(valid_fret_bool, min_length=MIN_FRET_LENGTH)
    valid_fret = trace.fret[:trace_lifetime][valid_fret_bool]
    if valid_fret.size > 0:
        mean_fret = np.mean(valid_fret)
        max_fret = np.max(valid_fret)
        first_fret = valid_fret[0]
        return (mean_fret, max_fret, first_fret)
    else:
        return (None, None, None)

def calculate_acceptor_lifetime(trace, trace_lifetime: int):
    """Calculates the acceptor lifetime of a smFRET trace."""
    valid_fret_bool = trace.fret[:trace_lifetime] >= MIN_FRET
    valid_fret_bool = run_length_filter(valid_fret_bool, min_length=MIN_FRET_LENGTH)
    acceptor_lifetime = np.sum(valid_fret_bool)
    return acceptor_lifetime

def count_fret_events(trace, trace_lifetime: int):
    """Counts the number of FRET events in a smFRET trace."""
    encoding = run_length_encode(trace.fret[:trace_lifetime] >= MIN_FRET)
    n_fret = np.sum(encoding == 1)  # Counts the end of each FRET
    return n_fret
 
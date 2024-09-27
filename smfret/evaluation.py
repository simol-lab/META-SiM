"""Library for running model evaluation."""
import numpy as np
import scipy


def get_dwell_time(states, target_state, target_destination_state=None):
    """Calculates the dwell times for a trace."""
    
    current_dwell_time = 0
    last_state = None
    fret_dwell_time = []
    is_first_dwell = True
    
    for state in states:
        if last_state is None:
            last_state = state
            current_dwell_time = 0
            continue

        if state == last_state:
            current_dwell_time += 1
        else:
            if is_first_dwell:
                is_first_dwell = False
            else:
                if last_state == target_state:
                    if target_destination_state is None:
                        fret_dwell_time.append(current_dwell_time)
                    elif state == target_destination_state:
                        fret_dwell_time.append(current_dwell_time)
            current_dwell_time = 0
            last_state = state
    return fret_dwell_time


def get_cdf(data, bins):
    """Calculates CDF of a data."""
    cdf = []
    data = np.array(data)
    n = len(data)
    for b in bins:
        cdf.append(np.sum(data <= b))
    cdf = np.array(cdf) / n
    return cdf


def dwell_time_cdf(x, k):
    """Returns ideal CDF funciton for exponential distribution."""
    return 1 - np.exp(-k * x)


def estimate_k(tau, bins, time_resolution, tau_min=0, return_fit_err=False):
    """Estimates the mean and std of kinetic constant k given dwell times."""
    effective_tau = [s - tau_min for s in tau if s >= tau_min]
    tau_cdf = get_cdf(effective_tau, bins)
    popt, pcov = scipy.optimize.curve_fit(dwell_time_cdf, bins * time_resolution, tau_cdf)
    perr = np.sqrt(np.diag(pcov))
    k = popt[0]
    if return_fit_err:
        return (k, perr[0])
    else:
        return k
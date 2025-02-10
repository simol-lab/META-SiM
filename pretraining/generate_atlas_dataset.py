from multiprocessing import Pool
from multiprocessing import RLock

import numpy as np
import os
from tqdm.auto import tqdm
import json
import re
import itertools
from pathlib import Path

from smfret.trace_simulator import Simulator
from smfret.trace_simulator import ParameterGenerator
from smfret.trace_simulator import SimulatedTraceSet

from smfret.multi_task_learning import FRETStateTraceSet
from smfret.dataset import FRETTraceSet

SIZE=1000
N_POOL=20


# naming the simulation conditions

# SNR
class SNR:
    c = (4, 8)  # clean
    n = (1.5, 4)  # noisy

# FRET value

class FRET_VALUE:
    l = (0.05, 0.35)  # low
    m = (0.4, 0.6)  # mid
    h = (0.65, 0.95)  # high

# kinetic rate (inverse)
class KINETIC_RATE:
    s = (40, 200)  # slow
    f = (5, 20)  # fast


def condition_to_param(n_state, snr, fret_value, kinetic_rate):
    snr_params = []
    fret_value_params = []
    kinetic_rate_params = []
    for s in snr:
        snr_params.append(getattr(SNR, s))
    for s in fret_value:
        fret_value_params.append(getattr(FRET_VALUE, s))
    for s in kinetic_rate:
        kinetic_rate_params.append(getattr(KINETIC_RATE, s))
    
    return int(n_state), snr_params, fret_value_params, kinetic_rate_params


def condition_to_name(n_state, snr, fret_value, kinetic_rate):
    return f"{n_state}-{snr}-{fret_value}-{kinetic_rate}"


def name_to_condition(name):
    n_state, snr, fret_value, kinetic_rate = name.split('-')
    return n_state, snr, fret_value, kinetic_rate


def generate_traceset(name):
    """Generates traceset for one class."""
    rng = np.random.default_rng()
    def make_transition_prob_fn(ks):
        counter = 0
        def transition_prob_fn():
            nonlocal counter
            counter += 1
            n = len(ks)
            return 1.0 / rng.uniform(low=ks[counter % n][0], high=ks[counter % n][1])
        return transition_prob_fn

    def make_fret_value_fn(Es):
        counter = -1
        def fret_value_fn():
            nonlocal counter
            counter += 1
            n = len(Es)
            return rng.uniform(low=Es[counter % n][0], high=Es[counter % n][1])
        return fret_value_fn
    
    param = condition_to_param(*name_to_condition(name))
    params_gen_base = ParameterGenerator(
        num_states_fn=lambda: param[0],
        snr_signal_fn=lambda: rng.uniform(*param[1][0]),
        snr_background_fn=lambda: rng.uniform(*param[1][0]),
        transition_prob_fn=make_transition_prob_fn(param[3]),
        fret_states_fn=make_fret_value_fn(param[2]),
        donor_lifetime_fn=lambda: rng.uniform(low=500, high=2000),
        trace_length_fn=lambda: 2000,
    )

    path = 'saved_dataset/'
    file = path + f'atlas_inference/{name}.npz'
    FRETStateTraceSet(size=SIZE, params_gen=params_gen_base).save(file)

if __name__ == '__main__':

    # 1 state
    names = []
    combinations = list(itertools.product([1], ['c', 'n'], ['l', 'm', 'h'], ['s']))
    for n_state, snr, fret_value, kinetic_rate in combinations:
        names.append(condition_to_name(n_state, snr, fret_value, kinetic_rate))
    
    # 2 state
    combinations = list(itertools.product([2], ['c', 'n'], ['lm', 'lh', 'mh'], ['s', 'f']))
    for n_state, snr, fret_value, kinetic_rate in combinations:
        names.append(condition_to_name(n_state, snr, fret_value, kinetic_rate))

    # 3 state
    combinations = list(itertools.product([3], ['c', 'n'], ['lmh'], ['s', 'f']))
    for n_state, snr, fret_value, kinetic_rate in combinations:
        names.append(condition_to_name(n_state, snr, fret_value, kinetic_rate))
    
    print('Total number of conditions =', len(names))
    with Pool(N_POOL, initargs=(RLock(),), initializer=tqdm.set_lock) as p:
        p.map(generate_traceset, names)
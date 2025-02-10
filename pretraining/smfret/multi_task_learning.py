"""Simulated smFRET traces dataset classes for multi-task learning."""

import numpy as np
from tqdm.auto import tqdm
import multiprocessing

import smfret.dataset
from smfret.dataset import MatlabTraceSet
from smfret.dataset import FRETTraceSet
from smfret.dataset import FRETTrace
from smfret.trace_simulator import Simulator
from smfret.trace_simulator import SingleChannelSimulator
from smfret.trace_simulator import ParameterGenerator
from smfret.trace_simulator import SimulatedTraceSet

import tensorflow as tf
from datetime import datetime as tm

rng = np.random.default_rng()

class EvolvingTraceSet(SimulatedTraceSet):
    """Class for trace sets capable of replacing X% traces with new traces."""
    def __init__(self, size, params_gen):
        super().__init__(None)
        self.size = 0
        self.params_gen = params_gen
        self.traces = []
        self.populate(size)
        self.vectorize_traces()
        self.is_labeled = True
    
    def populate(self, n: int):
        """Populates the dataset using the simulator."""
        new_traces = []
        try:
            pos = multiprocessing.current_process()._identity[0] - 1
            disable = False
        except:
            disable = True
            pos = 0
            
        desc_text = self.__class__.__name__
        for _ in tqdm(range(n), desc=desc_text, position=pos, disable=disable):
            new_traces.append(self.trace_gen())
        self.traces += new_traces
        self.size += n

    def delete(self, n: int):
        """Removes the traces on the top of the traces list."""
        self.traces = self.traces[n:]
        self.size -= n
        
    def evolve(self, n):
        """Adds new traces and removes old traces."""
        self.populate(n)
        self.delete(n)
        self.vectorize_traces()

    def vectorize_traces(self):
        """Stacks traces signals into np vectors."""
        self.time = self.traces[0].time
        self.donor = np.stack([trace.donor for trace in self.traces], axis=0)
        self.acceptor = np.stack([trace.acceptor for trace in self.traces], axis=0)
        self.label = np.stack([trace.label for trace in self.traces], axis=0)
    
    def trace_gen(self):
        """Generates a single labeled trace."""
        pass
    
    def save(self, file):
        """Saves the dataset to file."""
        self.vectorize_traces()
        data = {
            'time': self.time,
            'donor': self.donor,
            'acceptor': self.acceptor,
            'label': self.label,
            'size': self.size
        }
        np.savez(file, **data)
    

class SavedTraceSet(EvolvingTraceSet):
    """Class for trace sets saved on disk."""
    def __init__(self, size, file):
        self.file = file
        self.counter = 0
        self.epochs = 0
        self.cached_traces_data = None
        self.load_file_to_cache()
        super().__init__(size=size, params_gen=None)
        
    def resize(self, size):
        """Changes the size of the dataset."""
        self.counter = 0
        self.epochs = 0
        super().__init__(size, params_gen=None)
        
    def load_file_to_cache(self):
        """Reads the saved traces."""
        self.cached_traces_data = dict(np.load(self.file))
        total = self.cached_traces_data['donor'] + self.cached_traces_data['acceptor']
        max_total = np.expand_dims(total.max(axis=-1), axis=-1)
        self.cached_stacked = tf.cast(np.stack([
            self.cached_traces_data['donor'] / max_total,
            self.cached_traces_data['acceptor'] / max_total,
            total / max_total,
            np.repeat([self.cached_traces_data['time']], self.cached_traces_data['size'], axis=0) / smfret.dataset.TIME_DENOMINATOR
        ], axis=-1), tf.bfloat16)

        # reduce the memory footprint of cached_traces_data
        for key in self.cached_traces_data:
            if self.cached_traces_data[key].dtype == np.float64:
                self.cached_traces_data[key] = self.cached_traces_data[key].astype(np.float32, casting='same_kind')
        
    
    def trace_gen(self):
        trace = FRETTrace(
            donor=self.cached_traces_data['donor'][self.counter, :],
            acceptor=self.cached_traces_data['acceptor'][self.counter, :],
            time=self.cached_traces_data['time'],
            label=self.cached_traces_data['label'][self.counter, ...],
        )
        trace.counter = self.counter
        
        self.counter += 1
        if self.counter == self.cached_traces_data['size']:
            self.counter = 0
            self.epochs += 1
            
        return trace
    
    def vectorize_traces(self):
        """Overwrites the default behavior of vectorize_traces to gain performance."""
        with tf.device('/CPU:0'):
            self.time = self.traces[0].time
            counters = [trace.counter for trace in self.traces]
            self.donor = self.cached_traces_data['donor'][counters, :]
            self.acceptor = self.cached_traces_data['acceptor'][counters, :]
            self.label = self.cached_traces_data['label'][counters, ...]
            self.stacked = tf.gather(self.cached_stacked, counters)
        
    
    def to_tensor(self, size=None, normalize=True):
        """Converts the smFRET data into the form for training TensorFlow models."""
        with tf.device('/CPU:0'):
            return tf.cast(self.stacked, tf.float32)


class FRETStateTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with FRET states."""
    fret_states_resolution = 0.1
    lifetime_min = 500
    
    def trace_gen(self):
        trace = None
        while trace is None or trace.acceptor_lifetime <= self.lifetime_min:
            params = self.params_gen.generate()
            trace = Simulator(params).generate()

        trace.label = np.digitize(trace.fret_ideal, np.arange(0, 1, self.fret_states_resolution), right=True)
        return trace


class TwoStateQuickDynamicTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with quick FRET transitions."""
    fret_states_gap_min = 0.5
    lifetime_min = 500
    transition_prob_min = 0.03
    
    def trace_gen(self):
        trace = None
        while trace is None or trace.acceptor_lifetime <= self.lifetime_min:
            params = self.params_gen.generate()
            if params.num_states != 2:
                continue
            elif (np.max(params.fret_states) - np.min(params.fret_states)) < self.fret_states_gap_min:
                continue
            elif min(params.transition_prob_matrix[0, 1], params.transition_prob_matrix[1, 0]) < self.transition_prob_min:
                continue
            trace = Simulator(params).generate()
        return trace
    
class TwoStateQuickDynamicNBDTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with the N_(b+d)."""
    fret_states_resolution = 0.1
    fret_states_gap_min = 0.5
    lifetime_min = 500
    transition_prob_min = 0.05
    lifetime_min = 500
    quanta = 50
    lower_limit = 1
    upper_limit = 400
    quantize_bins = np.arange(lower_limit, upper_limit, quanta)
    
    def trace_gen(self):
        trace = None
        while trace is None or trace.acceptor_lifetime <= self.lifetime_min:
            params = self.params_gen.generate()
            if params.num_states != 2:
                continue
            elif (np.max(params.fret_states) - np.min(params.fret_states)) < self.fret_states_gap_min:
                continue
            elif min(params.transition_prob_matrix[0, 1], params.transition_prob_matrix[1, 0]) < self.transition_prob_min:
                continue
            trace = Simulator(params).generate()

        states = np.digitize(trace.fret_ideal, np.arange(0, 1, self.fret_states_resolution), right=True)
        nbd = np.sum(states[1:] != states[:-1])
        trace.label = np.digitize(nbd, self.quantize_bins, right=True)
        return trace


class TwoStateQuickDynamicTransitionProbTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with the transition probability."""
    fret_states_resolution = 0.1
    fret_states_gap_min = 0.5
    lifetime_min = 500
    transition_prob_min = 0.05
    lifetime_min = 500
    quanta = 4
    lower_limit = 1
    upper_limit = 20
    quantize_bins = np.arange(lower_limit, upper_limit, quanta)
    
    def trace_gen(self):
        trace = None
        while trace is None or trace.acceptor_lifetime <= self.lifetime_min:
            params = self.params_gen.generate()
            if params.num_states != 2:
                continue
            elif (np.max(params.fret_states) - np.min(params.fret_states)) < self.fret_states_gap_min:
                continue
            elif min(params.transition_prob_matrix[0, 1], params.transition_prob_matrix[1, 0]) < self.transition_prob_min:
                continue
            trace = Simulator(params).generate()
            
        if params.fret_states[0] > params.fret_states[1]:
            inverse_p = 1. /  params.transition_prob_matrix[0, 1]
        else:
            inverse_p = 1. / params.transition_prob_matrix[1, 0]

        trace.label = np.digitize(inverse_p, self.quantize_bins, right=True)
        return trace
    

class SelectedSegmentsLowFRETTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with selected segments."""
    lifetime_min = 500
    fret_state_max = 0.2
    
    def trace_gen(self):
        trace = None
        while trace is None or trace.acceptor_lifetime <= self.lifetime_min:
            params = self.params_gen.generate()
            if params.num_states != 1:
                continue
            elif np.min(params.fret_states) > self.fret_state_max:
                continue
            trace = Simulator(params).generate()
        return trace
    
    
class HighFRETBinaryTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with FRET states."""
    fret_states_resolution = 0.1
    lifetime_min = 500
    
    def trace_gen(self):
        trace = None
        while trace is None or trace.acceptor_lifetime <= self.lifetime_min:
            params = self.params_gen.generate()
            trace = Simulator(params).generate()

        states = np.digitize(trace.fret_ideal, np.arange(0, 1, self.fret_states_resolution), right=True)
        trace.label = np.array((states == np.max(states[:trace.acceptor_lifetime])))
        trace.label[trace.acceptor_lifetime:] = 0
        return trace
    
    
class LowFRETBinaryTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with FRET states."""
    fret_states_resolution = 0.1
    lifetime_min = 500
    
    def trace_gen(self):
        trace = None
        while trace is None or trace.acceptor_lifetime <= self.lifetime_min:
            params = self.params_gen.generate()
            trace = Simulator(params).generate()

        states = np.digitize(trace.fret_ideal, np.arange(0, 1, self.fret_states_resolution), right=True)
        trace.label = np.array((states == np.min(states[:trace.acceptor_lifetime])))
        trace.label[trace.acceptor_lifetime:] = 0
        return trace
    
    
class FRETStateHighestTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with FRET states."""
    fret_states_resolution = 0.1
    lifetime_min = 500
    
    def trace_gen(self):
        trace = None
        while trace is None or trace.acceptor_lifetime <= self.lifetime_min:
            params = self.params_gen.generate()
            trace = Simulator(params).generate()

        states = np.digitize(trace.fret_ideal, np.arange(0, 1, self.fret_states_resolution), right=True)
        trace.label = np.max(states[:trace.acceptor_lifetime])
        return trace


class FRETStateLowestTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with FRET states."""
    fret_states_resolution = 0.1
    lifetime_min = 500
    
    def trace_gen(self):
        trace = None
        while trace is None or trace.acceptor_lifetime <= self.lifetime_min:
            params = self.params_gen.generate()
            trace = Simulator(params).generate()

        states = np.digitize(trace.fret_ideal, np.arange(0, 1, self.fret_states_resolution), right=True)
        trace.label = np.min(states[:trace.acceptor_lifetime])
        return trace
    

class FRETStateCountTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with the number of FRET states."""
    fret_states_resolution = 0.1
    lifetime_min = 500
    quanta = 1
    lower_limit = 1
    upper_limit = 4
    quantize_bins = np.arange(lower_limit, upper_limit, quanta)
    
    def trace_gen(self):
        trace = None
        while trace is None or trace.acceptor_lifetime <= self.lifetime_min:
            params = self.params_gen.generate()
            trace = Simulator(params).generate()

        states = np.digitize(trace.fret_ideal, np.arange(0, 1, self.fret_states_resolution), right=True)
        trace.label = np.digitize(np.unique(states).size, self.quantize_bins, right=True)
        return trace
    
    
class NBDTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with the N_(b+d)."""
    fret_states_resolution = 0.1
    lifetime_min = 500
    quanta = 50
    lower_limit = 1
    upper_limit = 400
    quantize_bins = np.arange(lower_limit, upper_limit, quanta)
    
    def trace_gen(self):
        trace = None
        while trace is None or trace.acceptor_lifetime <= self.lifetime_min:
            params = self.params_gen.generate()
            trace = Simulator(params).generate()

        states = np.digitize(trace.fret_ideal, np.arange(0, 1, self.fret_states_resolution), right=True)
        nbd = np.sum(states[1:] != states[:-1])
        trace.label = np.digitize(nbd, self.quantize_bins, right=True)
        return trace
    
    
class DwellTimeTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with the average dwell time."""
    fret_states_resolution = 0.1
    lifetime_min = 500
    quanta = 20
    lower_limit = 10
    upper_limit = 200
    quantize_bins = np.arange(lower_limit, upper_limit, quanta)
    
    def trace_gen(self):
        trace = None
        while trace is None or trace.acceptor_lifetime <= self.lifetime_min:
            params = self.params_gen.generate()
            trace = Simulator(params).generate()
        
        states = np.digitize(trace.fret_ideal, np.arange(0, 1, self.fret_states_resolution), right=True)
        current_dwell_time = 0
        last_state = None
        dwell_times = []
        for state in states:
            if state == last_state:
                current_dwell_time += 1
            else:
                dwell_times.append(current_dwell_time)
                current_dwell_time = 1
                last_state = state
            if state == 0:
                break
        
        trace.label = np.digitize(np.mean(dwell_times), self.quantize_bins, right=True)
        return trace
    
    
class TransitionRateTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with the rate of state transitions."""
    fret_states_resolution = 0.2
    lifetime_min = 500
    quanta = np.log(2.0)
    lower_limit = np.log(10.0)
    upper_limit = np.log(200.0)
    quantize_bins = np.arange(lower_limit, upper_limit, quanta)
    eps = 1e-8
    
    def trace_gen(self):
        while True:
            params = self.params_gen.generate()
            fret_states = np.digitize(params.fret_states, np.arange(0, 1, self.fret_states_resolution), right=True)
            transition_out_rate = np.zeros_like(params.fret_states)
            
            if len(np.unique(fret_states)) <= 1:
                continue

            for i, fret in enumerate(params.fret_states):
                state = fret_states[i]
                for j, prob in enumerate(params.transition_prob_matrix[i, :]):
                    other_state = fret_states[j]
                    if other_state != state:
                        transition_out_rate[i] += prob
            
            avg_transition_time = np.mean(1.0 / (transition_out_rate + self.eps))
            log_transition_time = np.log(avg_transition_time)
            label = [log_transition_time]
            
            trace = Simulator(params).generate()

            if trace.acceptor_lifetime <= self.lifetime_min:
                continue
            else:
                trace.label = np.digitize(label, self.quantize_bins, right=True)
                return trace
            
            
class NoiseLevelTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with noise levels."""
    lifetime_min = 500
    quanta = 2
    lower_limit = 1
    upper_limit = 10
    quantize_bins = np.arange(lower_limit, upper_limit, quanta)
    
    def trace_gen(self):
        trace = None
        while trace is None or trace.acceptor_lifetime <= self.lifetime_min:
            params = self.params_gen.generate()
            trace = Simulator(params).generate()

        snr_signal = min(params.snr_signal, params.snr_background)
        trace.label = np.digitize(snr_signal, self.quantize_bins, right=True)
        return trace
    

class DonorLifetimeTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with donor lifetime."""
    quanta = 400.0
    lower_limit = 0.0
    upper_limit = 2000.0
    quantize_bins = np.arange(lower_limit, upper_limit, quanta)
    
    def trace_gen(self):
        params = self.params_gen.generate()
        trace = Simulator(params).generate()
        lifetime = trace.donor_lifetime
        trace.label = np.digitize(lifetime, self.quantize_bins, right=True)
        return trace
    
    
class AcceptorLifetimeTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with acceptor lifetime."""
    quanta = 400.0
    lower_limit = 0.0
    upper_limit = 2000.0
    quantize_bins = np.arange(lower_limit, upper_limit, quanta)
    
    def trace_gen(self):
        params = self.params_gen.generate()
        trace = Simulator(params).generate()
        lifetime = trace.acceptor_lifetime
        trace.label = np.digitize(lifetime, self.quantize_bins, right=True)
        return trace
    
    
class MultistepPhotobleachingTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with multistep photobleaching."""
    lifetime_min = 200
    quanta = 1.0
    lower_limit = 1.0
    upper_limit = 4.0
    quantize_bins = np.arange(lower_limit, upper_limit, quanta)
    
    def trace_gen(self):
        params = self.params_gen.generate()
        traces = []
        n_steps = rng.integers(self.lower_limit, self.upper_limit, endpoint=True)
        while len(traces) < n_steps:
            trace = Simulator(params).generate()
            lifetime = trace.acceptor_lifetime
            if lifetime is None or lifetime < self.lifetime_min:
                continue
            traces.append(trace)
        trace = traces[0]
        trace.donor = np.mean([x.donor for x in traces], axis=0)
        trace.acceptor = np.mean([x.acceptor for x in traces], axis=0)
        label = [n_steps]
        trace.label = np.digitize(label, self.quantize_bins, right=True)
        return trace
    
    
class SelectedSegmentsTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with selected segments."""
    lifetime_min = 500
    
    def trace_gen(self):
        params = self.params_gen.generate()
        while True:
            trace = Simulator(params).generate()
            lifetime = trace.donor_lifetime
            if lifetime is None or lifetime < self.lifetime_min:
                continue
            else:
                break
        return trace
    
    
class ReemergentTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with reemmergent trace segments."""
    lifetime_min = 200
    lifetime_max = 600
    
    quanta = 1.0
    lower_limit = 1.0
    upper_limit = 4.0
    quantize_bins = np.arange(lower_limit, upper_limit, quanta)
    
    gap_min = 200
    gap_max = 500
    
    def __init__(self, size, params_gen, binary_label=False):
        self.binary_label = binary_label
        super().__init__(size, params_gen)
    
    def trace_gen(self):
        params = self.params_gen.generate()
        traces = []
        n_traces = rng.integers(self.lower_limit, self.upper_limit, endpoint=True)
        
        while True:
            trace = Simulator(params).generate()
            lifetime = trace.acceptor_lifetime
            if lifetime is None or lifetime < self.lifetime_min or lifetime > self.lifetime_max:
                continue
            traces.append(trace)
            if len(traces) == n_traces:
                break
                
        combined_trace = None
        trace_length = traces[0].time.size
        
        for (i, trace) in enumerate(traces):
            # Concats all traces
            lifetime = trace.acceptor_lifetime
            if i < n_traces - 1:
                gap = rng.integers(self.gap_min, self.gap_max, endpoint=True)
                max_frame = lifetime + gap
            else:
                max_frame = None
            trace.donor = trace.donor[:max_frame]
            trace.acceptor = trace.acceptor[:max_frame]
            if self.binary_label:
                if i == 0:
                    trace.label = trace.label[:max_frame]
                else:
                    trace.label = 0 * trace.label[:max_frame]
            else:
                trace.label = (i + 1) * trace.label[:max_frame]
            
            if combined_trace is None:
                combined_trace = trace
            else:
                combined_trace.donor = np.concatenate([combined_trace.donor, trace.donor])
                combined_trace.acceptor = np.concatenate([combined_trace.acceptor, trace.acceptor])
                combined_trace.label = np.concatenate([combined_trace.label, trace.label])
        
        combined_trace.donor = combined_trace.donor[:trace_length]
        combined_trace.acceptor = combined_trace.acceptor[:trace_length]
        combined_trace.total = combined_trace.donor + combined_trace.acceptor
        combined_trace.label = combined_trace.label[:trace_length]
        combined_trace.time = np.arange(trace_length)    
        
        return combined_trace
    
    
class BackgroundTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with FRET background intensity."""
    resolution = 0.02
    background_clip = 0.2
    lifetime_min = 500
    
    def trace_gen(self):
        trace = None
        while trace is None or trace.acceptor_lifetime <= self.lifetime_min:
            params = self.params_gen.generate()
            trace = Simulator(params).generate()
        
        background = params.background / params.total_intensity
        states = np.digitize(background, np.arange(0, self.background_clip, self.resolution), right=True)
        trace.label = states
        return trace
    
    
class CoincidentTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with coincidentally overlapping donor and acceptor."""
    lifetime_min = 500
    
    def trace_gen(self):
        traces = []
        while len(traces) < 2:
            trace = None
            while trace is None or trace.acceptor_lifetime <= self.lifetime_min:
                params = self.params_gen.generate()
                trace = Simulator(params).generate()
            traces.append(trace)
        coincident_happened = rng.uniform(0, 1) > 0.5
        if coincident_happened:
            traces[0].donor = traces[1].donor  # replaces donor
        trace = traces[0]
        trace.label = coincident_happened
        return trace


class SingleChannelStateTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with FRET states."""
    fret_states_resolution = 0.1
    lifetime_min = 200
    
    def trace_gen(self):
        trace = None
        while trace is None or trace.acceptor_lifetime <= self.lifetime_min:
            params = self.params_gen.generate()
            trace = SingleChannelSimulator(params).generate()

        normalized_state = trace.acceptor_ideal / np.max(trace.acceptor_ideal)
        trace.label = np.digitize(normalized_state, np.arange(0, 1, self.fret_states_resolution), right=True)
        return trace


class SingleChannelStateFastTransitionTraceSet(SingleChannelStateTraceSet):
    """Class for trace sets labeled with FRET states for traces with fast transitions."""
    fret_states_resolution = 0.1
    transition_prob_increase_factor = 5
    snr_increase_factor = 2
    lifetime_min = 200

    def __init__(self, *args, **kwargs):
        """Overrides the parameter generator."""
        super().__init__(*args, **kwargs)
        self.params_gen = self.params_gen.copy()  # makes it safe to mutate the generator
        self.params_gen.transition_prob_fn = lambda: self.transition_prob_increase_factor * self.params_gen.transition_prob_fn()
        self.params_gen.snr_signal_fn = lambda: self.snr_increase_factor * self.params_gen.snr_signal_fn()
        self.params_gen.snr_background_fn = lambda: self.snr_increase_factor * self.params_gen.snr_background_fn()


class SingleChannelDwellTimeTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with the average dwell time."""
    fret_states_resolution = 0.2
    lifetime_min = 500
    quanta = 20
    lower_limit = 10
    upper_limit = 400
    quantize_bins = np.arange(lower_limit, upper_limit, quanta)
    
    def trace_gen(self):
        trace = None
        while trace is None or trace.acceptor_lifetime <= self.lifetime_min:
            params = self.params_gen.generate()
            trace = SingleChannelSimulator(params).generate()
        
        states = np.digitize(trace.fret_ideal, np.arange(0, 1, self.fret_states_resolution), right=True)
        
        current_dwell_time = 0
        last_state = None
        dwell_times = []
        for state in states:
            if state == last_state:
                current_dwell_time += 1
            else:
                dwell_times.append(current_dwell_time)
                current_dwell_time = 1
                last_state = state
            if state == 0:
                break
        
        trace.label = np.digitize(np.mean(dwell_times), self.quantize_bins, right=True)
        return trace


class SingleChannelBlinkingTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with FRET states."""
    lifetime_min = 200
    blinking_rate_increase_factor = 10
    
    def trace_gen(self):
        trace = None
        while trace is None or trace.acceptor_lifetime <= self.lifetime_min:
            params = self.params_gen.generate()
            params.non_cy3_blink_lifetime /= self.blinking_rate_increase_factor
            trace = SingleChannelSimulator(params).generate()

        trace.label = (trace.acceptor_ideal == 0) & (trace.time <= trace.acceptor_lifetime)
        return trace


class SingleChannelLifetimeTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with lifetime."""
    quanta = 200.0
    lower_limit = 0.0
    upper_limit = 2000.0
    quantize_bins = np.arange(lower_limit, upper_limit, quanta)
    
    def trace_gen(self):
        params = self.params_gen.generate()
        trace = SingleChannelSimulator(params).generate()
        lifetime = trace.acceptor_lifetime
        trace.label = np.digitize(lifetime, self.quantize_bins, right=True)
        return trace


class SingleChannelNoiseLevelTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with noise levels."""
    lifetime_min = 500
    quanta = 2
    lower_limit = 4
    upper_limit = 20
    quantize_bins = np.arange(lower_limit, upper_limit, quanta)

    def snr_increase_factor_fn(self):
        """Generates a random factor to increase the SNR."""
        return rng.uniform(1.0, 4.0)
    
    def trace_gen(self):
        trace = None
        while trace is None or trace.acceptor_lifetime <= self.lifetime_min:
            params = self.params_gen.generate()
            params.snr_signal *= self.snr_increase_factor_fn()
            params.snr_background *= self.snr_increase_factor_fn()
            trace = SingleChannelSimulator(params).generate()

        effective_snr = min(params.snr_signal, params.snr_background) / params.total_intensity * np.max(trace.acceptor_ideal)  # removes the contribution from donor
        trace.label = np.digitize(effective_snr, self.quantize_bins, right=True)
        return trace


class SingleChannelPhotobleachingStepsFrameLevelTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with photobleaching steps at the frame level."""
    num_steps_max = 6
    num_steps_min = 1
    lifetime_min = 200

    def num_steps_fn(self):
        """Generates the number of photobleaching steps."""
        return rng.integers(low=self.num_steps_min, high=self.num_steps_max, endpoint=True)


    def params_override(self, params):
        """Overrides params for this particular trace set."""
        params.num_states = 1
        params.initial_state = 0
        params.transition_prob_matrix = np.array([[1.0]])
        params.fret_states=[1.0]
        return params
    
    def trace_gen(self):

        traces = []
        num_steps = self.num_steps_fn()
        for _ in range(num_steps):
            trace = None
            while trace is None or trace.acceptor_lifetime <= self.lifetime_min:
                params = self.params_override(self.params_gen.generate())
                trace = SingleChannelSimulator(params).generate()

            trace.label = trace.time <= trace.acceptor_lifetime
            traces.append(trace)

        trace = traces[0]
        trace.acceptor = np.mean([x.acceptor for x in traces], axis=0)
        trace.acceptor_ideal = np.mean([x.acceptor_ideal for x in traces], axis=0)
        trace.label = np.sum([x.label for x in traces], axis=0)

        trace.label -= trace.label[-1]  # removes the unfinished steps from counting

        return trace


class SingleChannelPhotobleachingStepsTraceLevelTraceSet(SingleChannelPhotobleachingStepsFrameLevelTraceSet):
    """Class for trace sets labeled with photobleaching steps at the trace level."""
    def trace_gen(self):
        """Overrides the frame level trace gen."""
        trace = super().trace_gen()
        trace.label = trace.label[0] - trace.label[-1]
        return trace
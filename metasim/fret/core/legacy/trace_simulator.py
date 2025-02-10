"""Simulates smFRET traces."""

import numpy as np
import dataclasses
from typing import Callable
import logging

import smfret.dataset


DEFAULT_RNG = np.random.default_rng()

@dataclasses.dataclass
class SimulatorParameter:
    """Data class for parameters used in the simulator."""
    trace_length: int  # The total length of the generated trace
    num_states: int  # Number of FRET states
    fret_states: np.ndarray  # FRET states
    transition_prob_matrix: np.ndarray  # The transition probability matrix between FRET states
    initial_state: int  # The index of the initial state
    donor_lifetime: float  # Average donor lifetime before photobleaching
    acceptor_lifetime: float  # Average acceptor lifetime before photobleaching
    non_cy3_blink_lifetime: float  # Average contiguous trace length before a cy3 blinking
    cy3_blink_lifetime: float  # Average lengthof a cy3 blinking event
    total_intensity: float  # True total intensity of the smFRET trace
    background: float  # Average intensity of background.
    snr_background: float  # (total intensity + background) / std of background noise
    snr_signal: float  # (total intensity + background) / std of signal noise
    quantum_yield: float  # observed acceptor intensity / true acceptor intensity
    
    @property
    def donor_photobleach_prob(self):
        """Converts donor lifetime into donor photobleaching probability."""
        return 1.0 / self.donor_lifetime
    
    @property
    def acceptor_photobleach_prob(self):
        """Converts acceptor lifetime into acceptor photobleaching probability."""
        return 1.0 / self.acceptor_lifetime
    
    @property
    def cy3_blink_probability(self):
        """Converts Cy3 blink lifetime into Cy3 blink probability."""
        return 1.0 / self.non_cy3_blink_lifetime
    
    @property
    def donor_states(self):
        """Converts FRET states into donor intensities."""
        return [self.total_intensity * (1.0 - fret) for fret in self.fret_states] 
    
    @property
    def acceptor_states(self):
        """Converts FRET states into acceptor intensities."""
        return [self.total_intensity * (fret) for fret in self.fret_states] 
    
    @property
    def background_std(self):
        """Converts signal-to-noise ratio to std of background noise."""
        return (self.total_intensity + self.background) / self.snr_background
    
    @property
    def signal_std(self):
        """Converts SNR of signal to the std of signal."""
        return (self.total_intensity + self.background) / self.snr_signal
    

@dataclasses.dataclass
class ParameterGenerator:
    """Class for generating SimulatorParameter with given distributions."""
    trace_length_fn: Callable = lambda: 2000
    num_states_fn: Callable = lambda: DEFAULT_RNG.integers(low=1, high=5)
    fret_states_fn: Callable = lambda: DEFAULT_RNG.uniform(low=0.1, high=0.9)
    transition_prob_fn: Callable = lambda: DEFAULT_RNG.gamma(shape=5, scale=0.1)
    initial_state_fn: Callable = lambda fret_states: np.argmin(fret_states)
    donor_lifetime_fn: Callable = lambda: DEFAULT_RNG.uniform(low=800, high=1600)
    acceptor_lifetime_fn: Callable = lambda: DEFAULT_RNG.uniform(low=800, high=1600)
    non_cy3_blink_lifetime_fn: Callable = lambda: DEFAULT_RNG.uniform(low=200, high=1000)
    cy3_blink_lifetime_fn: Callable = lambda: DEFAULT_RNG.uniform(low=5, high=20)
    total_intensity_fn: Callable = lambda: DEFAULT_RNG.uniform(low=500, high=1500)
    background_fn: Callable = lambda: DEFAULT_RNG.uniform(low=10, high=100)
    snr_background_fn: Callable = lambda: DEFAULT_RNG.uniform(low=8, high=16)
    snr_signal_fn: Callable = lambda: DEFAULT_RNG.uniform(low=4, high=8)
    quantum_yield_fn: Callable = lambda: DEFAULT_RNG.uniform(low=0.95, high=1.0)
    
    def generate(self) -> SimulatorParameter:
        """Generates a SimulatorParameter object."""
        num_states = self.num_states_fn()
        fret_states = [self.fret_states_fn() for _ in range(num_states)]
        transition_prob_matrix = np.array(
            [[self.transition_prob_fn() for _ in range(num_states)] for _ in range(num_states)]
        )
        for i in range(num_states):
            transition_prob_matrix[i, i] += 1.0 - transition_prob_matrix[i, :].sum()
                
        return SimulatorParameter(
            trace_length=self.trace_length_fn(),
            num_states=num_states,
            fret_states=fret_states,
            transition_prob_matrix=transition_prob_matrix,
            initial_state=self.initial_state_fn(fret_states),
            donor_lifetime=self.donor_lifetime_fn(),
            acceptor_lifetime=self.acceptor_lifetime_fn(),
            non_cy3_blink_lifetime=self.non_cy3_blink_lifetime_fn(),
            cy3_blink_lifetime=self.cy3_blink_lifetime_fn(),
            total_intensity=self.total_intensity_fn(),
            background=self.background_fn(),
            snr_background=self.snr_background_fn(),
            snr_signal=self.snr_signal_fn(),
            quantum_yield=self.quantum_yield_fn(),
        )

    def copy(self):
        """Returns a copy of the same object."""
        return ParameterGenerator(dataclasses.asdict(self))
    
class SimulatedFRETTrace(smfret.dataset.FRETTrace):
    """Class for a simulated smFRET trace."""
    EPS = 1e-6
    def __init__(self, donor, acceptor, time, label, donor_ideal=None, acceptor_ideal=None, donor_lifetime=None, acceptor_lifetime=None):
        super().__init__(
            donor=donor,
            acceptor=acceptor,
            time=time,
            label=label,
        )
        self.donor_ideal = donor_ideal
        self.acceptor_ideal = acceptor_ideal
        self.donor_lifetime = donor_lifetime
        self.acceptor_lifetime = acceptor_lifetime
    
    @property
    def fret_ideal(self):
        """Ideal FRET values without noise."""
        return (
            self.acceptor_ideal / 
            (self.EPS + self.donor_ideal + self.acceptor_ideal)
        )
    
    @property 
    def total_ideal(self):
        """Ideal total intensity."""
        return self.donor_ideal + self.acceptor_ideal
        
    
class Simulator:
    """Base class for running smFRET simulations."""
    def __init__(self, parameter: SimulatorParameter):
        self.parameter = parameter
        self.rng = np.random.default_rng()
    
    def generate(self):
        """Generates a smFRET trace."""
        # Generic HMM simulation excluding Cy3 blinking and photobleaching
        size = self.parameter.trace_length
        n_states = self.parameter.num_states
        parameter = self.parameter
        state = np.zeros(size, dtype=np.int64)
        label = np.ones(size, dtype=np.int64)
        for i in range(size):
            if i == 0:
                state[i] = parameter.initial_state
            else:
                state[i] = self.rng.choice(
                    range(n_states),
                    p=self.parameter.transition_prob_matrix[state[i - 1], :],
                )
        donor = np.take(np.array(self.parameter.donor_states), state)
        acceptor = np.take(np.array(self.parameter.acceptor_states), state)
        
        # Store ideal donor and acceptor intensities
        donor_ideal = np.array(donor)
        acceptor_ideal = np.array(acceptor)
        
        # Quantum yield effects
        acceptor *= parameter.quantum_yield
        
        # Noise effects for signals
        effective_std = np.sqrt(
            max(0, parameter.signal_std**2 - parameter.background_std**2)
        )
        donor_noise_signal =  self.rng.normal(
            loc=0, scale=(np.sqrt(0.5) * effective_std), size=size)
        donor += donor_noise_signal  # Equal std in two channels is a good simple assumption
        acceptor_noise_signal = self.rng.normal(
            loc=0, scale=(np.sqrt(0.5) * effective_std), size=size)
        acceptor += acceptor_noise_signal
        
        # Photobleach effects
        donor_bleach_time = np.int64(self.rng.exponential(scale=parameter.donor_lifetime))
        acceptor_bleach_time = np.int64(self.rng.exponential(scale=parameter.acceptor_lifetime))
        if acceptor_bleach_time < size:
            donor[acceptor_bleach_time:] += acceptor[acceptor_bleach_time:]
            acceptor[acceptor_bleach_time:] = 0
            label[acceptor_bleach_time:] = 0
            donor_ideal[acceptor_bleach_time:] += acceptor_ideal[acceptor_bleach_time:]
            acceptor_ideal[acceptor_bleach_time:] = 0
        if donor_bleach_time < size:
            donor[donor_bleach_time:] = 0
            acceptor[donor_bleach_time:] = 0
            label[donor_bleach_time:] = 0
            donor_ideal[donor_bleach_time:] = 0
            acceptor_ideal[donor_bleach_time:] = 0
        
        
        # Cy3 blinking effects
        effective_size = min(donor_bleach_time, size)
        blink_frame = np.int64(self.rng.exponential(scale=parameter.non_cy3_blink_lifetime))
        while blink_frame < effective_size:
            blink_lifetime = np.int64(self.rng.exponential(scale=parameter.cy3_blink_lifetime))
            if blink_lifetime > 0:
                blink_end_frame = min(effective_size, blink_frame + blink_lifetime)
                donor[blink_frame:blink_end_frame] = 0
                acceptor[blink_frame:blink_end_frame] = 0
                label[blink_frame:blink_end_frame] = 0
                donor_ideal[blink_frame:blink_end_frame] = 0
                acceptor_ideal[blink_frame:blink_end_frame] = 0
            blink_frame += blink_lifetime
            blink_frame += np.int64(self.rng.exponential(scale=parameter.non_cy3_blink_lifetime))
        
        # Noise effects for background
        donor_noise_background = self.rng.normal(
            loc=0.5 * parameter.background,
            scale=(np.sqrt(0.5) * parameter.background_std),
            size=size,
        )   # Equal std in two channels is a good simple assumption
        donor += donor_noise_background 
        acceptor_noise_background = self.rng.normal(
            loc=0.5 * parameter.background,
            scale=(np.sqrt(0.5) * parameter.background_std),
            size=size,
        )
        acceptor += acceptor_noise_background
        
        donor_bleach_time = min(donor_bleach_time, size)
        acceptor_bleach_time = min(acceptor_bleach_time, donor_bleach_time)
        
        time = np.array(list(range(1, size + 1)))
        trace = SimulatedFRETTrace(
            donor=donor,
            acceptor=acceptor,
            time=time,
            label=label,
            donor_ideal=donor_ideal,
            acceptor_ideal=acceptor_ideal,
            donor_lifetime=donor_bleach_time,
            acceptor_lifetime=acceptor_bleach_time,
        )
        return trace


class SingleChannelSimulator(Simulator):
    """Class for a simulator of single channel data."""
    EPS = 1e-7
    def generate(self):
        """overrides the two-channel generating logic."""
        trace = super().generate()
        trace.donor *= self.EPS
        return trace


class SimulatedTraceSet(smfret.dataset.FRETTraceSet):
    """Class for storing simulated smFRET traces."""
    def __init__(self, simulator: Simulator):
        super().__init__()
        self.simulator = simulator
    
    def populate(self, n: int):
        """Populates the dataset using the simulator."""
        self.size = n
        self.traces = [self.simulator.generate() for _ in range(n)]
        self.time = self.traces[0].time
        self.donor = np.stack([trace.donor for trace in self.traces], axis=0)
        self.acceptor = np.stack([trace.acceptor for trace in self.traces], axis=0)
        self.label = np.stack([trace.label for trace in self.traces], axis=0)
        self.is_labeled = True
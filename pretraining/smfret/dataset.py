import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import tensorflow as tf
import smfret.trace_statistics

TRACE_KEY = 'traces'
DONOR_TO_ACCEPTOR_CORRECTION = 0.09
TIME_DENOMINATOR = 2000
FRET_EFFECTIVE_RANGE = (-0.2, 1.2)


class FRETTrace:
    """The base class for a smFRET trace."""
    def __init__(self, donor, acceptor, time, label):
        self.donor = donor
        self.acceptor = acceptor
        self.total = self.donor + self.acceptor
        self.time = time
        self.label = label
        self.statistics = None

    @property
    def fret(self):
        """Getter function for calculating the FRET value."""
        corrected_acceptor = self.acceptor - DONOR_TO_ACCEPTOR_CORRECTION * self.donor
        fret = (corrected_acceptor) / (self.donor + corrected_acceptor + np.finfo(np.float32).eps)
        return fret
    
    def analyze(self):
        self.statistics = smfret.trace_statistics.analyze_trace(self)

        
class FRETTraceSet:
    """The base class for a dataset to store traces."""
    def __init__(self):
        self.original_file = None
        self.time = None
        self.donor = None
        self.acceptor = None
        self.size = None
        self.label = None
        self.is_labeled = None
        self._cached_stacked_time = None
        self.traces = []

    def plot_traces(self, n_start=0, n_traces=10, selected_traces_only=False):
        """Plots samples of stored traces in the dataset."""
        i = n_start - 1
        count = 0
        while count < n_traces and i < (self.size - 1):
            i += 1
            if selected_traces_only and not any(self.label[i]):
                continue
            count += 1
            plt.subplot(n_traces, 1, count)
            plt.plot(self.time, self.donor[i, :], color='tab:blue', linewidth=0.5)
            plt.plot(self.time, self.acceptor[i, :], color='tab:red', linewidth=0.5)
            
            start_end_frames = []
            max_line = max(np.max(self.donor[i, :]), np.max(self.acceptor[i, :]))
            if len(self.label[i, :]) == len(self.donor[i, :]):
                for j in range(self.time.shape[0]):
                    last_frame_label = self.label[i, j - 1] if j > 0 else 0
                    if np.logical_xor(last_frame_label, self.label[i, j]):
                        plt.axvline(x=self.time[j], color='tab:green', linestyle='-.')
                        start_end_frames.append(j)
                        if len(start_end_frames) == 2:
                            plt.plot(
                                start_end_frames,
                                [max_line, max_line],
                                color='tab:green',
                                linewidth=2,
                            )
                            start_end_frames = []
                if len(start_end_frames) == 1:
                    # Selection continues to the last frame.
                    plt.axvline(x=self.time[-1], color='tab:green', linestyle='-.')
                    start_end_frames.append(len(self.time) - 1)
                    plt.plot(
                        start_end_frames,
                        [max_line, max_line],
                        color='tab:green',
                        linewidth=2,
                    )
                    
            self.traces[i].analyze()
            if self.traces[i].statistics.trace_lifetime is not None:
                plt.axvline(x=self.traces[i].statistics.trace_lifetime, color='tab:orange', linestyle=':')
            if self.traces[i].statistics.mean_donor is not None:
                plt.axhline(y=self.traces[i].statistics.mean_donor, color='tab:blue', linestyle=':')
            if self.traces[i].statistics.mean_acceptor is not None:
                plt.axhline(y=self.traces[i].statistics.mean_acceptor, color='tab:red', linestyle=':')
            
            plt.annotate(f'trace {i}', (len(self.time) / 2.0, 0.9))
            plt.ylabel('Intensity (A.U.)')
        plt.xlabel('Frame')
        plt.tight_layout()

    def __str__(self):
        string = f"""FRET traces dataset with {self.size} traces.
size = {self.size}
length = {self.time.shape[0]}
is_labeled = {self.is_labeled}
        """
        return string

    def __repr__(self):
        return self.__str__()
    
    def __add__(self, other):
        return CombinedTraceSet(self, other)
    
    def fret_prob_distribution(self, bins=28, predicted_label=None):
        """Calculates the prob distribution of FRET values."""
        if predicted_label is None:
            idx = np.where(self.label)
        else:
            idx = np.where(predicted_label)
        fret = self.fret[idx]
        prob, bins = np.histogram(fret, bins=bins, range=FRET_EFFECTIVE_RANGE, density=True)
        return prob * (bins[1] - bins[0]), bins
    
    def valid_fret_values(self, predicted_label=None):
        """Calculates the FRET values of selected segments."""
        if predicted_label is None:
            idx = np.where(self.label)
        else:
            idx = np.where(predicted_label)
        valid_fret = self.fret[idx]
        cleaned_fret = valid_fret[(valid_fret > FRET_EFFECTIVE_RANGE[0]) 
                                  & (valid_fret < FRET_EFFECTIVE_RANGE[1])]
        return cleaned_fret

    def plot_fret_histogram_of_selected_segments(self, bins=28, predicted_label=None):
        """Plots the histogram of FRET value for selected segments."""
        idx = np.where(self.label)
        fret = self.fret[idx]
        plt.hist(fret, bins=bins, range=FRET_EFFECTIVE_RANGE, alpha=0.8, facecolor='tab:blue', edgecolor='k', linewidth=1.5, label='Ground Truth', density=True)
        if predicted_label is not None:
            idx = np.where(predicted_label)
            fret = self.fret[idx]
            plt.hist(fret, bins=bins, range=FRET_EFFECTIVE_RANGE, alpha=0.8, facecolor='tab:orange', edgecolor='k', linewidth=1.5, label='Prediction', density=True)
        plt.legend()
    
    def to_tensor(self, size=None, normalize=True):
        """Converts the smFRET data into the form for training TensorFlow models."""
        donor = tf.cast(self.donor, tf.float32)
        acceptor = tf.cast(self.acceptor, tf.float32)
        total = donor + acceptor
        tensor = tf.stack([donor, acceptor, total], axis=-1)
        total_max = tf.reduce_max(total, axis=(-1))
        
        if normalize:
            total_max = tf.expand_dims(tf.expand_dims(total_max, axis=-1), axis=-1)
            tensor = tensor / total_max
        
        if self._cached_stacked_time is None:
            time = tf.cast(tf.repeat([self.time], self.size, axis=0), dtype=tf.float32)
            time = tf.expand_dims(time / tf.cast(TIME_DENOMINATOR, time.dtype), axis=-1)
            self._cached_stacked_time = time
        else:
            time = self._cached_stacked_time
        
        tensor = tf.concat([tensor, time], axis=-1)
        return tensor
    
    @property
    def fret(self):
        """Getter function for calculating the FRET value."""
        corrected_acceptor = self.acceptor - DONOR_TO_ACCEPTOR_CORRECTION * self.donor
        fret = (corrected_acceptor) / (self.donor + corrected_acceptor + np.finfo(np.float32).eps)
        return fret
    
    @property
    def entire_trace_label(self):
        """Getter function for converting frame-wise labels into trace-wise labels."""
        trace_label = np.sum(self.label, axis=-1) > 0
        return trace_label
    
    def plot_fret_jump_heatmap(self, n_frames=1):
        """Plots a heatmap of the FRET values for current and n frames before."""
        plt.hist2d(
            self.fret[:, :-n_frames].flatten(),
            self.fret[:, n_frames:].flatten(),
            bins=50,
            range=((0, 1), (0, 1)),
            norm=matplotlib.colors.LogNorm(),
            cmap='plasma',
        )
        plt.colorbar()
        plt.xlabel('FRET after')
        plt.ylabel('FRET before')
    
    def broadcast_data_to_traces(self):
        """Broadcasts the data store in arrays into individual trace objects."""
        self.traces = []
        for i in range(self.size):
            trace = FRETTrace(
                time=self.time,
                donor=self.donor[i, :],
                acceptor=self.acceptor[i, :],
                label=self.label[i, :],
            )
            self.traces.append(trace)
        
    def analyze(self):
        """Runs the analyze() method for each smFRET trace."""
        for trace in self.traces:
            trace.analyze()
 
    def statistics_to_tensor(self):
        """Converts the statistics of each trace to a dense Tensor."""
        return np.array([trace.statistics for trace in self.traces])
    
    def copy(self, trace_set):
        """Copied data from another trace set."""
        self.original_file = trace_set.original_file
        self.time = trace_set.time
        self.donor = trace_set.donor
        self.acceptor = trace_set.acceptor
        self.size = trace_set.size
        self.label = trace_set.label
        self.is_labeled = trace_set.is_labeled
        self.traces = trace_set.traces
        
    def trim(self, n_frame, start_frame=0):
        """Shortens each trace to a fixed number of frames."""
        self.donor = self.donor[:, start_frame : start_frame + n_frame]
        self.acceptor = self.acceptor[:, start_frame : start_frame + n_frame]
        self.label = self.label[:, start_frame : start_frame + n_frame]
        self.time = self.time[:self.donor.shape[1]]
        self.broadcast_data_to_traces()
        
    def train_test_split(self, train_percentage):
        """Splits the trace set into two randomly selected sets."""
        idx = np.random.choice([True, False], size=self.size, p=[train_percentage, 1.0 - train_percentage])
        
        train_set = FRETTraceSet()
        train_set.original_file = self.original_file
        train_set.time = self.time
        train_set.donor = self.donor[idx, :]
        train_set.acceptor = self.acceptor[idx, :]
        train_set.size = np.sum(idx)
        train_set.label = self.label[idx, :]
        train_set.is_labeled = self.is_labeled
        train_set.broadcast_data_to_traces()
        
        test_set = FRETTraceSet()
        test_set.original_file = self.original_file
        test_set.time = self.time
        test_set.donor = self.donor[~idx, :]
        test_set.acceptor = self.acceptor[~idx, :]
        test_set.size = self.size - np.sum(idx)
        test_set.label = self.label[~idx, :]
        test_set.is_labeled = self.is_labeled
        test_set.broadcast_data_to_traces()
        
        return (train_set, test_set)

    def get_subset(self, indices):
        """Gets a subset of the trace set."""
        new_set = FRETTraceSet()
        new_set.original_file = self.original_file
        new_set.time = self.time
        new_set.donor = self.donor[indices, :]
        new_set.acceptor = self.acceptor[indices, :]
        new_set.size = new_set.donor.shape[0]
        new_set.label = self.label[indices, :]
        new_set.is_labeled = self.is_labeled
        new_set.broadcast_data_to_traces()
        return new_set
        
            

class MatlabTraceSet(FRETTraceSet):
    def __init__(self, file: str):
        super().__init__()
        matlab_data = scipy.io.loadmat(file)[TRACE_KEY]
        self.file = file
        self.original_file = matlab_data['file'][0, 0][0]
        self.time = matlab_data['time'][0, 0][0].astype(np.int64)
        self.donor = matlab_data['donor'][0, 0].astype(np.int64)
        self.acceptor = matlab_data['acceptor'][0, 0].astype(np.int64)
        self.size = matlab_data['count'][0, 0][0, 0].astype('int')
        self.label = matlab_data['label'][0, 0].astype(np.int64)
        self.is_labeled = matlab_data['islabeled'][0, 0][0, 0].astype('bool')
        self.broadcast_data_to_traces()
        

class CombinedTraceSet(FRETTraceSet):
    def __init__(self, set_a, set_b):
        super().__init__()
        if not np.array_equal(set_a.time, set_b.time):
            #TODO(leyou): Use padding when combining two datasets of different time length.
            raise ValueError("The time axis of two trace sets much be the same.")
        
        self.time = set_a.time
        self.donor = np.concatenate((set_a.donor, set_b.donor), axis=0)
        self.acceptor = np.concatenate((set_a.acceptor, set_b.acceptor), axis=0)
        self.size = set_a.size + set_b.size
        self.label = np.concatenate((set_a.label, set_b.label), axis=0)
        self.is_labeled = set_a.is_labeled or set_b.is_labeled
        self.broadcast_data_to_traces()
        
        
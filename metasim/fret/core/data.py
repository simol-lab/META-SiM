"""Library for data structures in META-SiM."""
from openfret import Trace
from openfret import Dataset

TIME_DENOMINATOR = 2000   # A constant rescaler used in training


def _trace_data_to_tensor(donor, acceptor):
    """Helper function to convert data into tensor."""
    import tensorflow as tf
    if acceptor is None:
        raise ValueError('Acceptor cannot be set to None.')
    acceptor = tf.cast(acceptor, tf.float32)
    if donor is None:
        # Fill in zeros for single color scenario.
        # In training, acceptor is used to host the single channel data.
        donor = tf.zeros_like(acceptor)
    else:
        donor = tf.cast(donor, tf.float32)
    total = donor + acceptor
    tensor = tf.stack([donor, acceptor, total], axis=-1)
    total_max = tf.reduce_max(total, axis=(-1))
    total_max = tf.expand_dims(total_max, axis=-1)
    tensor = tensor / total_max
    time = tf.expand_dims((tf.range(tensor.shape[0], dtype=tf.float32) + 1.0) / TIME_DENOMINATOR, axis=-1)
    tensor = tf.concat([tensor, time], axis=-1)
    return tensor


class OneColorDataset(Dataset):
    """Class for single color dataset."""
    def __init__(self, dataset, channel_type):
        self.__dict__.update(dataset.__dict__)  # copies from parent class
        self.channel_type = channel_type

    def trace_to_tensor(self, trace: Trace):
        """Converts a trace into a TensorFlow tensor."""
        signal = None
        for channel in trace.channels:
            if channel.channel_type == self.channel_type:
                signal = channel.data
        if signal is None:
            raise ValueError(f"{self.channel_type} channel must exist in all traces.")
        return _trace_data_to_tensor(donor=None, acceptor=signal)

    def to_tensors(self):
        """Converts a dataset into a list of tensors."""
        import tensorflow as tf
        tensor_list = []
        for trace in self.traces:
            with tf.device('/CPU:0'):   # reduces communication overhead.
                tensor_list.append(self.trace_to_tensor(trace))
        return tensor_list


class TwoColorDataset(Dataset):
    """Class for two color FRET dataset."""
    def __init__(self, dataset, donor_channel_type="donor", acceptor_channel_type="acceptor"):
        self.__dict__.update(dataset.__dict__)  # copies from parent class
        self.donor_channel_type = donor_channel_type
        self.acceptor_channel_type = acceptor_channel_type

    def trace_to_tensor(self, trace: Trace):
        """Converts a trace into a TensorFlow tensor."""
        donor = None
        acceptor = None
        for channel in trace.channels:
            if channel.channel_type == self.donor_channel_type:
                donor = channel.data
            if channel.channel_type == self.acceptor_channel_type:
                acceptor = channel.data
        if donor is None:
            raise ValueError(f"{self.donor_channel_type} channel must exist in all traces.")
        if acceptor is None:
            raise ValueError(f"{self.acceptor_channel_type} channel must exist in all traces.")
        return _trace_data_to_tensor(donor, acceptor)

    def to_tensors(self):
        """Converts traces into a list of tensors."""
        import tensorflow as tf
        tensor_list = []
        for trace in self.traces:
            with tf.device('/CPU:0'):
                # Places this operation on CPU to reduce communication overhead.
                tensor_list.append(self.trace_to_tensor(trace))
        return tensor_list


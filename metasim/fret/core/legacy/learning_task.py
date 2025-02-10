"""Library for defining the learning tasks used in multi-task learning."""

import numpy as np
import tensorflow as tf
from typing import Callable

import smfret.dataset


class Task:
    """Base class for defining learning tasks."""
    feature_tensor: tf.Tensor = None
    labels:tf.Tensor = None
    def get_mini_batch(self, size):
        """Gets a mini batch of training data and labels."""
        ...
    def loss(self, y_pred, y_true) -> tf.Tensor:
        """Calculates the loss given input data and labels."""
        ...
    def predict(self, input_tensor) -> tf.Tensor:
        """Runs the inference given the input data."""
        ...
        
class SimpleTask(Task):
    """A simple task class that only envolve one feature tensor and one label tensor."""
    def __init__(self, feature_tensor, labels, model=None, name=None, shuffle=True):
        self.name = name  # A human readable name for readability
        self.num_traces = feature_tensor.shape[0]
        self.feature_tensor = feature_tensor
        self.labels = labels
        self.model = model
        self.shuffle = shuffle
        self.batch_counter = 0
        self.epochs = 0
        self.run_shuffle()

    def run_shuffle(self):
        """Runs the shuffle for the order of how a mini batch of data is read."""
        if self.shuffle:
            self.shuffle_order = np.random.permutation(list(range(self.num_traces)))
        else:
            self.shuffle_order = list(range(self.num_traces))

    def get_mini_batch(self, size):
        """Gets a mini batch of data."""
        n = self.shuffle_order[self.batch_counter]
        self.batch_counter += 1
        if self.batch_counter >= self.num_traces:
            self.batch_counter = 0
            self.epochs += 1
            self.run_shuffle()
            
        if n + size > self.num_traces:  # Exceeds the upper bound of the feature tensor
            return self.get_mini_batch(size)  # Finds the next valid return

        batch_tensor = self.feature_tensor[n : n + size, ...]
        batch_labels = self.labels[n : n + size, ...]
        
        return (batch_tensor, batch_labels)
    
    def train(self, epochs=1, **kwargs):
        """Trains the model."""
        return self.model.fit(x=self.feature_tensor, y=self.labels, shuffle=self.shuffle, epochs=epochs, **kwargs)
        
        
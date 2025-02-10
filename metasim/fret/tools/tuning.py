"""Library for downstream tasks using META-SiM."""

import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Union, Optional, Any
from pathlib import Path
from sklearn.linear_model import LogisticRegression

VERSION = 'METASIM-1.0.0'  # First version of META-SiM

 # Group training tasks according to area of FRET feature.
TASKS = [
     ['DWELLTIME', 'TRANSITIONPROBQUICK', 'NBDQUICK', 'NBD', 'TRANSITIONRATE'],
     ['FRETSTATECOUNT', ],
     ['FRETSTATEHIGHEST', 'FRETSTATELOWEST'],
     ['MULTISTEP', 'COINCIDENCE',],
     ['ACCEPTORLIFETIME'],
     ['DONORLIFETIME'],
     ['NOISELEVEL', 'BACKGROUND'],
     ['SINGLECHANNELDWELLTIME', 'SINGLECHANNELLIFETIME',],
     ['SINGLECHANNELPHOTOBLEACHINGSTEPSTRACE',],
     ['SINGLECHANNELNOISELEVEL',],
]

NAMES = [
     'Kinetic\nRate',
     'FRET State\nNumber',
     'FRET\nValue',
     'Photobleaching\nSteps',
     'Acceptor\nLifetime',
     'Donor\nLifetime',
     'Noise',
     'Single Ch.\nKinetic Rate',
     'Single Ch.\nPhotobleaching',
     'Single Ch.\nNoise',
 ]

@dataclass
class Weights:
    kernel: List[List[float]]
    bias: List[float]

    def to_dict(self):
        return {"kernel": self.kernel, "bias": self.bias}

    @classmethod
    def from_dict(cls, data):
        if isinstance(data["kernel"], np.ndarray):
            data["kernel"] = data["kernel"].tolist()
        if isinstance(data["bias"], np.ndarray):
            data["bias"] = data["bias"].tolist()
        return cls(kernel=data["kernel"], bias=data["bias"])


@dataclass
class TuningSetting:
    id: int
    optimizer: str
    learningRate: float
    batchSize: int
    epochs: int
    validationSplit: float
    l2Regularization: float

    def to_dict(self):
        return {
            "id": self.id,
            "optimizer": self.optimizer,
            "learningRate": self.learningRate,
            "batchSize": self.batchSize,
            "epochs": self.epochs,
            "validationSplit": self.validationSplit,
            "l2Regularization": self.l2Regularization,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data["id"],
            optimizer=data["optimizer"],
            learningRate=data["learningRate"],
            batchSize=data["batchSize"],
            epochs=data["epochs"],
            validationSplit=data["validationSplit"],
            l2Regularization=data["l2Regularization"],
        )


@dataclass
class ModelConfig:
    weights: Weights
    lookupList: List[Any]
    version: Optional[str] = VERSION
    tuningSetting: Optional[TuningSetting] = None

    def to_dict(self):
        if isinstance(self.lookupList, np.ndarray):
            self.lookupList = self.lookupList.tolist()
        for i, e in enumerate(self.lookupList):
            if getattr(e, 'dtype', None) == np.int64:
                self.lookupList[i] = int(self.lookupList[i])
        data = {
            "weights": self.weights.to_dict(),
            "lookupList": self.lookupList,
            "version": self.version,
        }
        if self.tuningSetting:
            data["tuningSetting"] = self.tuningSetting.to_dict()
        return data

    @classmethod
    def from_dict(cls, data):
        weights = Weights.from_dict(data["weights"])
        tuning_setting = TuningSetting.from_dict(data["tuningSetting"]) if "tuningSetting" in data else None
        return cls(weights=weights, lookupList=data["lookupList"], version=data["version"], tuningSetting=tuning_setting)

    def save_to_json(self, file_path: Path):
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4) # indent makes it readable

    @classmethod
    def load_from_json(cls, file_path: Path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def load_from_task(cls, task: str):
        """Loads pre-trained downstream tasks."""
        file_path = os.path.join(
            os.path.dirname(__file__),
            'downstream_task_model_configs',
            f'{task}.json',
        )
        return cls.load_from_json(file_path)

    def predict(self, embedding):
        """Generates predicted labels."""
        logits = np.matmul(
            embedding, self.weights.kernel) + self.weights.bias
        idx = np.argmax(logits, -1)
        pred = np.array(self.lookupList)[idx]
        return pred

    def predict_proba(self, embedding):
        """Generates predicted probability distribution."""
        logits = np.matmul(
            embedding, self.weights.kernel) + self.weights.bias
        pred = softmax(logits)
        return pred

    def predict_regression(self, embedding, optimize_for='RMSE'):
        """Generates regression prediction for continuous labels.

        The output is decided by the final target.
        - If optimize for MAE, predict the median of predicted distribution
        - If optimize for RMSE, predict the expectation of distribution

        Args:
            embedding: The embedding numpy tensor.
            optimize_for: 'RMSE' or 'MAE'.
        """
        if optimize_for.upper() not in ['MAE', 'RMSE']:
            raise ValueError(
                f'Regression can only optimize for MAE or RMSE, got {optimize_for}',
            )
        if optimize_for.upper() == 'RMSE':
            pred = np.einsum(
                'ij,j->i',
                self.predict_proba(embedding),
                np.array(self.lookupList),
            )
            return pred
        elif optimize_for.upper() == 'MAE':
            cum_prob = np.cumsum(self.predict_proba(embedding), axis=-1)
            idx = np.argmax(cum_prob >= 0.5, axis=-1)
            return np.array(self.lookupList)[idx]


def softmax(x):
    """
    Numerically stable softmax function for a 2D array.

    Args:
        x: A 2D numpy array.

    Returns:
        A 2D numpy array with the softmax applied along the last axis.
    """

    # Find the maximum value along the last axis (axis=1) for each row.  This is crucial for numerical stability.
    max_x = np.max(x, axis=1, keepdims=True)  # keepdims=True is important for proper broadcasting

    # Subtract the maximum value from x before exponentiating. This prevents potential overflow issues.
    shifted_x = x - max_x

    # Exponentiate the shifted values.
    exp_x = np.exp(shifted_x)

    # Calculate the sum of exponentials along the last axis.
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)

    # Normalize by dividing by the sum of exponentials.
    softmax_x = exp_x / sum_exp_x

    return softmax_x


def train_classification(
        embedding, label, class_weight=None, max_iter=5000,
):
    """Trains a classification head for downstream task.

    This function uses logistic regression to train a head for trace
    level classification task.

    Args:
        embedding: The numpy tensor of trace embeddings.
        label: The numpy array of trace labels.
        class_weight: An optional dict to contain weights for each class.
        max_iter: An optional integer to define max iterations for optimization.

    Returns:
        A ModelConfig object for the trained task head.
    """
    clf = LogisticRegression(
        max_iter=max_iter, tol=1e-6, class_weight=class_weight,
    ).fit(embedding, label)
    if clf.coef_.shape[0] == 1:
        clf.coef_ = np.concatenate([np.zeros_like(clf.coef_), clf.       coef_], axis=0)
        clf.intercept_ = np.concatenate([np.zeros_like(clf.intercept_),  clf.intercept_], axis=0)
    num_classes = clf.coef_.shape[0]
    weights = Weights(
        kernel=clf.coef_.T.tolist(),
        bias=clf.intercept_.tolist(),
    )
    model = ModelConfig(
        weights=weights,
        lookupList=clf.classes_.tolist(),
    )
    return model


def train_regression(
        embedding, label, max_iter=5000, num_bins=8,
):
    """Trains a regression model.

    This function trains a regression model. In META-SiM, this is done via
    quantization, i.e. bucketing labels into classes.

    The output is decided by the final target. We use:
        - If optimize for MAE, the median of predicted distribution
        - If optimize for RMSE, the expectation of predicted distribution

    Args:
        embedding: The embedding numpy tensor.
        label: The numpy array of label.
        max_iter: The max iterations of optimization.
        num_bins: The number of bins for quantization.

    Return:
        A nmpy array of prediction.
    """
    lower_limit = np.min(label)
    upper_limit = np.max(label)
    quanta = (upper_limit - lower_limit) / num_bins
    quantize_bins = np.arange(lower_limit, upper_limit, quanta)
    quantized_label = np.digitize(label, quantize_bins, right=True)
    model = train_classification(
        embedding, quantized_label, max_iter=max_iter,
    )
    model.lookupList = (quantize_bins + 0.5 * quanta).tolist()
    model.lookupList[-1] -= 0.5 * quanta  # caps the max
    model.lookupList = [lower_limit] + model.lookupList  # right is (a, b].
    return model


def moving_majority_vote(arr, window_size):
    """
    Applies a moving majority vote filter to a 1D numpy array.

    Args:
        arr: The input 1D numpy array.
        window_size: The size of the moving window. Must be an odd integer.

    Returns:
        A numpy array of the same size as the input, containing the majority vote
        at each position.  Returns the original array if window_size is not a
        positive odd integer or if window_size is greater than the array size.
    """

    if not isinstance(window_size, int) or window_size <= 0 or window_size % 2 == 0 or window_size > len(arr):
      return arr  # Return original array if window size is invalid

    n = len(arr)
    result = np.zeros(n, dtype=arr.dtype)  # Ensure output has the same data type

    half_window = window_size // 2

    for i in range(n):
        window_start = max(0, i - half_window)
        window_end = min(n, i + half_window + 1)  # +1 because range is exclusive of end

        window = arr[window_start:window_end]

        if len(window) == 0:  # Handle edge cases where the window is empty.
            result[i] = arr[i] # or some default value.
            continue


        counts = np.bincount(window) # efficient counting of occurrences

        if len(counts) == 0: # Handle edge cases where the window is empty.
            result[i] = arr[i] # or some default value.
            continue

        majority_index = np.argmax(counts) # index of the most frequent value
        result[i] = majority_index

    return result

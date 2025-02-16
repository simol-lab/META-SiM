"""Library for FRET idealization using META-SiM."""

import os
import numpy as np
import metasim.fret.core.data as core_data
import metasim.fret.core.model as core_model


CHECKPOINT_PATH = 'FRETSTATE.h5'
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), CHECKPOINT_PATH)
MODEL_RESOLUTION = 0.1


def idealize(dataset):
    """Generates idealized FRET intensities for a dataset."""
    model = core_model.Model(checkpoint_path=CHECKPOINT_PATH)
    pred = model.encode_frames(dataset)
    idealized_states = [
        np.argmax(p, axis=-1) * MODEL_RESOLUTION for p in pred
    ]
    # Reduce half of resolution for non-zero FRET states to offset
    # quantization error
    idealized_states = [
        p - (p > 0) * 0.5 * MODEL_RESOLUTION for p in idealized_states
    ]
    return idealized_states


def get_fret_efficiency(dataset, idealized_states):
    """Gets the list of FRET efficiency values.

    In this function, we use state idealization to identify active FRET regions.
    Activate region is defined as state > 0, per training definition.

    Args:
        dataset: A dataset object.
        idealized_states: A list of numpy array of FRET state.

    Returns:
        A list of numpy array of FRET efficiencies. This list can be used to construct FRET histogram.
    """
    tensors = dataset.to_tensors()
    efficiency = []
    for i, tensor in enumerate(tensors):
        states = idealized_states[i]
        fret = tensor[..., 1] / (tensor[..., 0] + tensor[..., 1])
        values = fret[states > 0]
        efficiency.append(np.array(values))
    return efficiency


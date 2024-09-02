# META-SiM

## Fine-tuning Library for Pretrained META-SiM Models
### **Introduction**
This library provides functions for fine-tuning pretrained models for classification tasks using TensorFlow and Keras.

### Functions

* **`encode_traces(trace_set, encoder)`**

    - Encodes a set of traces into a NumPy matrix using the provided encoder.
    - Takes two arguments:
        - `trace_set`: A custom object representing a set of traces for encoding.
        - `encoder`: A pre-trained model (e.g., TensorFlow model) that takes traces as input and outputs encoded representations.
    - Returns a NumPy matrix containing the encoded representations of the traces.

* **`finetune_classification_head(encoder, train_sets, epochs, num_classes=2, trace_level=True, lr=1e-3, verbose=0, return_embedding=False, use_sklearn=False, embedding_list=None, l2_regularization=0.0)`**

    - Fine-tunes a classification head on top of a pre-trained encoder for a multi-class classification task.
    - Takes several arguments:
        - `encoder`: The pre-trained model for encoding traces.
        - `train_sets`: A list of custom objects representing the training data containing trace sets and labels.
        - `epochs`: The number of epochs for training the classification head (default: 100).
        - `num_classes` (optional): The number of classes in the classification task (default: 2).
        - `trace_level` (optional): Boolean indicating whether training is done on trace-level or frame-level (default: True).
        - `lr` (optional): Learning rate for the optimizer (default: 1e-3).
        - `verbose` (optional): Verbosity level for training (default: 0).
        - `return_embedding` (optional): Boolean indicating whether to return the encoded representations (default: False).
        - `use_sklearn` (optional): Boolean indicating whether to use scikit-learn's LogisticRegression for logistic regression (default: False).
        - `embedding_list` (optional): A list of pre-computed encoded representations if `use_sklearn` is `False` (default: None).
        - `l2_regularization` (optional): L2 regularization coefficient for the classification head (default: 0.0).
    - Returns:
        - If `return_embedding` is `True`: A tuple containing the trained classification model (`classifier`) and a list of encoded representations (`embedding_list`).
        - If `return_embedding` is `False`: The trained classification model (`classifier`).


## Library for Model Evaluation

### **Introduction**

This Python library provides functions for model evaluation, specifically focusing on dwell time analysis for state transitions. The library includes functions to:

- Calculate dwell times for a given trace
- Compute the cumulative distribution function (CDF) of dwell times
- Estimate the kinetic constant (k) based on the dwell time distribution

### **Functions**

#### **1. `get_dwell_time(states, target_state, target_destination_state=None)`**
* **Purpose:** Calculates the dwell times for a given trace.
* **Parameters:**
  - `states`: A list of states in the trace.
  - `target_state`: The target state for which dwell times are calculated.
  - `target_destination_state`: (Optional) If specified, only dwell times that transition to this state are considered.
* **Returns:** A list of dwell times for the target state.

#### **2. `get_cdf(data, bins)`**
* **Purpose:** Calculates the CDF of a given data set.
* **Parameters:**
  - `data`: A list or array of data points.
  - `bins`: A list of bin edges for the CDF calculation.
* **Returns:** A list of CDF values corresponding to the bins.

#### **3. `dwell_time_cdf(x, k)`**
* **Purpose:** Returns the ideal CDF function for an exponential distribution.
* **Parameters:**
  - `x`: The x-values for the CDF.
  - `k`: The rate parameter of the exponential distribution.
* **Returns:** The CDF values for the exponential distribution.

#### **4. `estimate_k(tau, bins, time_resolution, tau_min=0, return_fit_err=False)`**
* **Purpose:** Estimates the kinetic constant (k) based on the dwell times.
* **Parameters:**
  - `tau`: A list of dwell times.
  - `bins`: A list of bin edges for the CDF calculation.
  - `time_resolution`: The time resolution of the data.
  - `tau_min`: (Optional) The minimum dwell time to consider.
  - `return_fit_err`: (Optional) If True, returns the estimated k and its fitting error.
* **Returns:** The estimated kinetic constant (k), and optionally its fitting error.

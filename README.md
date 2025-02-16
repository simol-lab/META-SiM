# META-SiM: A Foundation Model for Efficient Biological Discovery in Single-Molecule Time Traces

Single-molecule fluorescence microscopy (SMFM) is a powerful tool for revealing rare biological intermediates, but the resulting data often requires time-consuming manual inspection, hindering systematic analysis.  We introduce META-SiM, a transformer-based foundation model designed to accelerate biological discovery from SMFM data. Pre-trained on diverse SMFM analysis tasks, META-SiM excels at various analyses, including trace selection, classification, segmentation, idealization, and photobleaching analysis.

Beyond individual trace analysis, META-SiM generates high-dimensional embedding vectors for each trace, enabling efficient whole-dataset visualization, labeling, comparison, and sharing.  Combined with the objective metric of Local Shannon Entropy, this visualization facilitates rapid identification of condition-specific behaviors, even subtle or rare ones.  Application of META-SiM to existing smFRET data revealed a previously unobserved intermediate state in pre-mRNA splicing, demonstrating its potential to remove bottlenecks, improve objectivity, and accelerate biological discovery in complex single-molecule data.

# The `matesim` Python Library

The `matesim` Python library provides a user-friendly interface for leveraging the power of the META-SiM foundation model.  It offers tools for data loading, processing, embedding generation, visualization, and building machine learning models for classification and regression tasks.

`matesim` is particularly well-suited for researchers working with FRET data who want to:

* Generate embeddings using the META-SiM model.
* Visualize embeddings using UMAP and smFRET Atlas.
* Discover condition-specific traces using Local Shannon Entropy.
* Train and evaluate machine learning models for time-trace analysis.
* Evaluate the consistency of human labels for time traces.

## Installation

Install `matesim` using pip:

```bash
pip install matesim
```

## Getting Started

This example demonstrates a basic workflow for building a classification model using META-SiM embeddings:

```python
import openfret
import matesim
import numpy as np

# Load data using the OpenFRET library.
data = matesim.fret.data.TwoColorDataset(
    openfret.read_data("<path_to_your_openfret_data>")
)

# Load the pre-trained META-SiM model.
model = matesim.fret.Model()

# Generate embeddings for the time traces.
embeddings = model(data)

# Prepare labels for supervised fine-tuning.
labels = np.array([trace.metadata["your_label_name"] for trace in data.traces])

# Train a task-specific classification model.
task_model = matesim.fret.tuning.train_classification(
    embeddings, labels
)

# Generate predictions using the trained model.
predictions = task_model.predict(embeddings)
```

The example below shows how to create Atlas and data-specific UMAPs using META-SiM:

```python
import openfret
import matesim
import numpy as np

# Load data using the OpenFRET library.
data = matesim.fret.data.TwoColorDataset(
    openfret.read_data("<path_to_your_openfret_data>")
)

# Load the pre-trained META-SiM model.
model = matesim.fret.Model()

# Generate embeddings for the time traces.
embeddings = model(data)

# Prepare labels for supervised fine-tuning.
labels = np.array([trace.metadata["your_label_name"] for trace in data.traces])

# Get the local Shannon Entropy
entropy = metasim.fret.get_entropy(
    embedding,
    label,
)

# Plot the smFRET Atlas with data
metasim.fret.tools.viz.plot_atlas(
    embedding=embedding,
    label=label,
    color=entropy,
    color_name='Entropy',
)

# Plot the data-specific UMAP
reducer = metasim.fret.tools.viz.get_umap_reducer(embedding)
umap_coord = reducer.transform(embedding)
metasim.fret.tools.viz.plot_umap(
    umap_coord=umap_coord,
    label=label,
    color=entropy,
    color_name='Entropy',
);

# Plot the FRET histograms
idealized_states = metasim.fret.tools.idealize.idealize(dataset)

efficiency = metasim.fret.tools.idealize.get_fret_efficiency(
    dataset,
    idealized_states,
)

metasim.fret.tools.viz.plot_fret_histograms(
    efficiency,
    label,
)
```

For more detailed examples and tutorials, please refer to the documentation and examples available in the `matesim` repository.

## Contributing

Contributions to `matesim` are welcome! Please see the repository for more information.

## License

`matesim` is licensed under the MIT License.

## Citation

Li J, Zhang L, Johnson-Buck A, Walter NG. Foundation model for efficient biological discovery in single-molecule data. Res Sq [Preprint]. 2024 Oct 17:rs.3.rs-4970585/v1. doi: 10.21203/rs.3.rs-4970585/v1. PMID: 39483892; PMCID: PMC11527229.

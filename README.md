# META-SiM Foundataion Model Python Library

META-SiM is a foundation model for single-molecule time traces. It is published in paper:

Foundation model for efficient biological discovery in single-molecule data
https://www.biorxiv.org/content/10.1101/2024.08.26.609721v1

This Python library provides tools for reading and writing single-molecule FRET (Förster Resonance Energy Transfer) data according to the OpenFRET data format specification (version 1.0.0).

## Installation

```bash
pip install openfret
```

## Usage

The library provides classes for representing the different components of the OpenFRET data format: `Dataset`, `Trace`, `Channel`, and `Metadata`.

### Writing Data

```python
from openfret import Dataset, Trace, Channel, Metadata, write_openfret
from datetime import date

# Create Channel objects
channel1 = Channel("donor", [10.0, 12.0, 15.0], excitation_wavelength=488.0, emission_wavelength=520.0)
channel2 = Channel("acceptor", [2.0, 5.0, 8.0], excitation_wavelength=532.0, emission_wavelength=580.0)

# Create a Trace object
trace1 = Trace([channel1, channel2], metadata=Metadata({"molecule_id": "1"}))

# Create a Dataset object
dataset = Dataset(
    title="My FRET Experiment",
    traces=[trace1],
    description="FRET data of protein folding",
    experiment_type="2-Color FRET",
    authors=["John Doe", "Jane Smith"],
    institution="University X",
    date=date(2024, 1, 1),
    metadata=Metadata({"experiment_id": "123"}),
    sample_details={"buffer_conditions": "Phosphate buffer", "other_details": Metadata({"ph": 7.4})}, #Example of nested metadata
    instrument_details={"microscope": "Olympus IX71", "other_details": Metadata({"objective": "60x"})}, #Example of nested metadata
)

# Write the dataset to a JSON file
write_openfret(dataset, "fret_data.json")
```

### Reading Data

```python
from openfret import read_openfret

# Read the dataset from a JSON file
loaded_dataset = read_openfret("fret_data.json")

# Access data
print(loaded_dataset.title)
for trace in loaded_dataset.traces:
    for channel in trace.channels:
        print(f"Channel type: {channel.channel_type}, Data: {channel.data}")
print(loaded_dataset.sample_details["other_details"]) #Accessing the nested metadata
print(loaded_dataset.instrument_details["other_details"]) #Accessing the nested metadata

# Or print the whole dictionary
import json
print(json.dumps(loaded_dataset.to_dict(), indent=4, default=str)) #added default=str to handle the date object in json output
```

## Classes

*   **`Metadata(dict)`:** Represents user-defined metadata. Inherits from `dict` to allow arbitrary key-value pairs.
*   **`Channel`:** Represents a single data channel with intensity data and associated metadata.
    *   `channel_type` (str): String identifier for the channel (e.g., donor, acceptor, FRET).
    *   `excitation_wavelength` (float, optional): Excitation wavelength in nanometers.
    *   `emission_wavelength` (float, optional): Emission wavelength in nanometers.
    *   `exposure_time` (float, optional): Exposure time per frame in seconds.
    *   `data` (List[float]): Intensity values for the channel.
    *   `metadata` (`Metadata`, optional): Metadata for the channel.
*   **`Trace`:** Represents a single-molecule trace containing multiple channels.
    *   `channels` (List[`Channel`]): Array of channel data for this trace.
    *   `metadata` (`Metadata`, optional): Metadata for the trace.
*   **`Dataset`:** Represents a collection of single-molecule traces.
    *   `title` (str): Title of the experiment or dataset.
    *   `description` (str, optional): Detailed description of the experiment.
    *   `experiment_type` (str, optional): Type of experiment.
    *   `authors` (List[str], optional): List of authors.
    *   `institution` (str, optional): Institution where the experiment was conducted.
    *   `date` (`datetime.date`, optional): Date of the experiment.
    *   `traces` (List[`Trace`]): Array of single-molecule traces.
    *   `metadata` (`Metadata`, optional): Metadata for the dataset.
    *   `sample_details` (dict, optional): Details about the sample.
    *   `instrument_details` (dict, optional): Details about the instrument.

## Functions

*   **`write_openfret(dataset: Dataset, filename: str)`:** Writes a `Dataset` object to a JSON file.
*   **`read_openfret(filename: str) -> Dataset`:** Reads a `Dataset` object from a JSON file.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

MIT License

## Acknowledgements

This library is based on the OpenFRET data format specification.

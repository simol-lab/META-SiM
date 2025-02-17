# Tutorials for META-SiM

In the tutorial section, we will use META-SiM for the functionalities demonstrated in the paper:

Li J, Zhang L, Johnson-Buck A, Walter NG. Foundation model for efficient biological discovery in single-molecule data. Res Sq [Preprint]. 2024 Oct 17:rs.3.rs-4970585. doi: 10.21203/rs.3.rs-4970585/v1. PMID: 39483892; PMCID: PMC11527229.

To run these tutorials, you are recommended to use Google Colab. Here are the links to open the tutorials in a Google Colab environment:

* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simol-lab/META-SiM/blob/main/tutorials/metasim_umap_and_entropy.ipynb) Use META-SiM for visualizing and clustering trace embeddings and discover condition specific behaviors.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simol-lab/META-SiM/blob/main/tutorials/metasim_classification.ipynb) Use META-SiM for trace and frame classification tasks with fine-tuning. 
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simol-lab/META-SiM/blob/main/tutorials/metasim_regression.ipynb) Use META-SiM for trace regression tasks with fine-tuning.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simol-lab/META-SiM/blob/main/tutorials/matasim_single_color.ipynb) Use META-SiM for a classification task with single-color intensity v.s. time dataset.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simol-lab/META-SiM/blob/main/tutorials/metasim_built_in_tasks.ipynb) Use META-SiM's built-in tasks for data analysis.


To run these tutorials on your local environment, you are recommended to create a virtual environment and install metasim with

```bash
python3 -m pip install metasim
```

# Use your Own Data

To use your own data for data analysis, we recommend using the OpenFRET Python library to convert your data into a standard data format used by META-SiM. To learn more about the data conversion, check out the [OpenFRET Python Library](https://github.com/simol-lab/OpenFRET/tree/main/python).

# Exploring Directed vs. Undirected Graph-Based Topological Data Analysis of Transformer Attention Maps

This project investigates the application of directed graph analysis to transformer attention maps, comparing it with traditional undirected graph methods.

## Course Project: Selected Topics in DS 2024 Course

### Dataset Used
The dataset used for this project is the IMDb Movie Reviews dataset. This dataset consists of 50,000 movie reviews from IMDb, split evenly into 25,000 training and 25,000 testing sets. Each review is labeled as either positive or negative, making it a binary sentiment classification task. The IMDb dataset is a standard benchmark for natural language processing tasks, particularly for sentiment analysis.

Key Features of the Dataset:
- **Size:** 50,000 reviews (25,000 for training, 25,000 for testing)
- **Labels:** Binary sentiment classification (positive/negative)
- **Source:** IMDb movie reviews

For more details, you can refer to the original dataset [here](https://ai.stanford.edu/~amaas/data/sentiment/).

### Reproducibility
All experiments were performed in Google Colab. Links to all experiments:
- [Experiment 1: TDA_comparison_1layer_1head.ipynb](https://colab.research.google.com/drive/1jVAllgl7QsgoYiYv5Id3Pz3e6RCNmb9G?usp=sharing)
- [Experiment 2: TDA_comparison_all_layers_and_heads.ipynb](https://colab.research.google.com/drive/19r0_7g4pRpUdv1GTHkn__DTieQl2l8zr?usp=sharing)
- [Experiment 3: directed_features_1layer_1head.ipynb](https://colab.research.google.com/drive/1H8VjiWcSdVE08Wi2PhuqJbUO6p3pX6Sb?usp=sharing)
- [Experiment 4: directed_features_all_layers_and_heads.ipynb](https://colab.research.google.com/drive/1jyG41UcMQMtcfoApTWt7TDgaRhZKeBwE?usp=sharing)

### Description of Notebooks

- **TDA_comparison_1layer_1head.ipynb**  
  Training 3 models on features obtained from the last layer and the last head of attention: baseline, baseline + directed TDA, and baseline + ubdirected TDA.

- **TDA_comparison_all_layers_and_heads.ipynb**  
  Training 3 models on features obtained from all layers and all heads of attention: baseline, baseline + directed TDA, and baseline + undirected TDA.

- **directed_features_1layer_1head.ipynb**  
  Calculation of TDA-features received from the last layer and the last attention head, considering attention maps as both directed and undirected graphs.

- **directed_features_all_layers_and_heads.ipynb**  
  Calculation of TDA-features received from all layers and all heads of attention, considering attention maps as both directed and undirected graphs.

### Description of Scripts

- **all_layers_heads_features.py**  
  A script for calculating the TDA-features obtained from all layers and all heads of attention, considering attention maps as both directed and undirected graphs. Used in `directed_features_all_layers_and_heads.ipynb`.

- **directed_graphs_TDA.py**  
  A script for calculating TDA-features received from the last layer and the last attention head, considering attention maps as both directed and undirected graphs. Used in `TDA_comparison_1layer_1head.ipynb`.

- **baseline_train.py**  
  Script for training the baseline model.

- **directed_TDA_train.py**  
  Script for training models: baseline + directed TDA, baseline + undirected TDA.




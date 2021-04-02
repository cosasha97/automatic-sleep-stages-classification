# automatic-sleep-stages-classification

This repository presents the work of Sasha Collin and Clément Nguyen for the roject for the class of 'Machine Learning for Time-Series' - Master MVA

## Generate results
In the jupyter notebook 'sleep_stages_classification.ipynb' is shown step by step how to generate results similar to the ones presented in our report.


## Content
The repository contains the following folders/files:
- script/: contains all the scripts used to compute features, perform relevance analysis or clustering
  - atoms.py: functions and methods used for Convolutional Dictionnary Learning (CDL). Many functions are inspired/taken from the TP2 done in class with Charles Truong.
  - automatic_features_generation.py: functions and methods used to generate custom features. Some functions are adapted from the TP1 done in class with Charles Truong.
  - features.py: functions used to generate entropy features
  - generate_features.py: pipeline run in practice to generate entropy features
  - utils.py: useful functions and classes, including the class DataLoader (to load and preprocess raw EEG data), the functions to perform features relevance analysis or clustering
- data/: containing some saved features to plot more quickly results on notebooks
- report.pdf: our report summurizing our work on the automatic sleep stages classification
- sleep_stages_classification.ipynb: see the 'Generate results' section above

# automatic-sleep-stages-classification

This repository presents the work of Sasha Collin and Clément Nguyen for the roject for the class of 'Machine Learning for Time-Series' - Master MVA

## Context
Sleep has a very important biological role and has been related to plasticity and reorganization of memory. Unfortunately, sleep disorders are a very common issue that affect a significant number of people worldwide, and have a serious impact on people's health. That's why trying to better understand how sleep works and trying to detect and characterize sleep stages is a hot research topic.

There are 5 different sleep stages: wake (W), drowsiness (N1), light sleep (N2), deep sleep (N3), and rapid eye movement sleep (REM). The classification of such stages, when done manually, is difficult, time consuming, and can lead to annotation errors. To tackle this issue, the authors of \cite{rodriguez2014automatic} propose an unsupervised classification scheme for sleep stage prediction, using entropy features computed from electroencephalograms (EEGs), features relevance analysis, and unsupervised clustering.

We implemented from scratch this algorithm and, to assess the relevance of entropy features, we designed custom features and compared the performances of the algorithm on both sets of features. Finally, we implemented a semi-supervised classification method based on Convolutional Dictionary Learning and compared the classification performances of both algorithms.

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
- sleep_stages_classification.ipynb: see the 'Generate results' section below


## Generate results
In the jupyter notebook 'sleep_stages_classification.ipynb' is shown step by step how to generate results similar to the ones presented in our report.

# This script computes and saves the entropy features.

import numpy as np
import pandas as pd
import os
import mne
import time

from features import *
from utils import *


def save_dict_to_csv(features, output_path, feature_names):
    df_temp = pd.DataFrame.from_dict(features)
    df_temp[feature_names] = df_temp[feature_names].astype('float32')
    df_temp[['id', 'night', 'ann_id', 'label']] = df_temp[['id', 'night', 'ann_id', 'label']].astype('int8')
    df_temp['time'] = df_temp['time'].astype('int32')
    df_temp.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)


def init_feature_dict(data, start, end):
    features = {}
    features['id'] = [data['id'] for k_ in range(end - start + 1)]
    features['night'] = [data['night'] for k_ in range(end - start + 1)]
    features['time'] = list(data['times'])[start: end + 1]
    features['ann_id'] = [data['ann_id'] for k_ in range(end - start + 1)]
    features['label'] = list(data['labels'])[start: end + 1]
    return features

feature_names = [
    'FD-Fpz',
    'FD-Pz',
    'HFD-Fpz',
    'HFD-Pz',
    'KFD-Fpz',
    'KFD-Pz',
    'DFA-a1-Fpz',
    'DFA-a2-Fpz',
    'DFA-a1-Pz',
    'DFA-a2-Pz',
    'H-Fpz',
    'H-Pz',
    'ApEn-m1-Fpz',
    'ApEn-m2-Fpz',
    'ApEn-m1-Pz',
    'ApEn-m2-Pz',
    'SampEn-m1-Fpz',
    'SampEn-m2-Fpz',
    'SampEn-m1-Pz',
    'SampEn-m2-Pz',
    'MSE-tau2-Fpz',
    'MSE-tau3-Fpz',
    'MSE-tau4-Fpz',
    'MSE-tau5-Fpz',
    'MSE-tau6-Fpz',
    'MSE-tau2-Pz',
    'MSE-tau3-Pz',
    'MSE-tau4-Pz',
    'MSE-tau5-Pz',
    'MSE-tau6-Pz'
]

if __name__ == '__main__':

    output_path = 'features.csv'
    resume = os.path.exists(output_path)
    if resume:
        resume_csv = pd.read_csv(output_path)
        resume_csv = resume_csv[['id', 'night', 'time']]

    save_every = 100

    dataloader = DataLoader()

    for k, data in enumerate(dataloader.get_data()):

        eeg_fpz_all = data['data'][:, 0, :]
        eeg_pz_all = data['data'][:, 1, :]

        eeg_fpz_mean = eeg_fpz_all.reshape(-1).mean()
        eeg_fpz_std = eeg_fpz_all.reshape(-1).std()
        eeg_pz_mean = eeg_pz_all.reshape(-1).mean()
        eeg_pz_std = eeg_pz_all.reshape(-1).std()

        num_epochs = len(data['labels'])

        if resume:
            if sum((resume_csv['id'] == data['id']) & (resume_csv['night'] == data['night'])) == num_epochs:
                continue

        features = {}
        for feat_name in feature_names:
            features[feat_name] = []

        for i in range(num_epochs):

            if resume:
                if sum((resume_csv['id'] == data['id']) & (resume_csv['night'] == data['night']) &
                       (resume_csv['time'] == data['times'][i])) == 1:
                    continue

            time0 = time.time()
            print('Sample {} -- Epoch {} / {}'.format(k + 1, i + 1, num_epochs))

            eeg_fpz = eeg_fpz_all[i]
            eeg_pz = eeg_pz_all[i]

            # Normalization
            eeg_fpz = (eeg_fpz - eeg_fpz_mean) / eeg_fpz_std
            eeg_pz = (eeg_pz - eeg_pz_mean) / eeg_pz_std

            # Fractal dimension
            features['FD-Fpz'].append(fractal_dimension(eeg_fpz))
            features['FD-Pz'].append(fractal_dimension(eeg_pz))

            features['HFD-Fpz'].append(higuchi_fractal_dimension(eeg_fpz))
            features['HFD-Pz'].append(higuchi_fractal_dimension(eeg_pz))

            features['KFD-Fpz'].append(katz_fractal_dimension(eeg_fpz))
            features['KFD-Pz'].append(katz_fractal_dimension(eeg_pz))

            # Detrended Fluctuation Analysis
            features['DFA-a1-Fpz'].append(dfa(eeg_fpz, 4, 16))
            features['DFA-a2-Fpz'].append(dfa(eeg_fpz, 16, 64))
            features['DFA-a1-Pz'].append(dfa(eeg_pz, 4, 16))
            features['DFA-a2-Pz'].append(dfa(eeg_pz, 16, 64))

            # Shannon Entropy
            features['H-Fpz'].append(shannon_entropy(eeg_fpz))
            features['H-Pz'].append(shannon_entropy(eeg_pz))

            # Approximate Entropy & Sample Entropy
            sampen, apen = fast_sampen_apen(eeg_fpz, 3)
            features['ApEn-m1-Fpz'].append(apen[0])
            features['ApEn-m2-Fpz'].append(apen[1])
            features['SampEn-m1-Fpz'].append(sampen[0])
            features['SampEn-m2-Fpz'].append(sampen[1])

            sampen, apen = fast_sampen_apen(eeg_pz, 3)
            features['ApEn-m1-Pz'].append(apen[0])
            features['ApEn-m2-Pz'].append(apen[1])
            features['SampEn-m1-Pz'].append(sampen[0])
            features['SampEn-m2-Pz'].append(sampen[1])

            # Multiscale Entropy
            taus = [2, 3, 4, 5, 6]
            for tau in taus:
                features['MSE-tau{}-Fpz'.format(tau)].append(multiscale_entropy(eeg_fpz, 2, tau))
                features['MSE-tau{}-Pz'.format(tau)].append(multiscale_entropy(eeg_pz, 2, tau))

        temp_features = init_feature_dict(data, 0, num_epochs - 1)
        for key in features.keys():
            temp_features[key] = features[key]
        save_dict_to_csv(temp_features, output_path, feature_names)


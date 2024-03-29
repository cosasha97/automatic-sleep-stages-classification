import os
import mne
import itertools
import numpy as np
from tqdm.notebook import tqdm
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


class DataLoader:
    """
    Framework to load EEG data.
    """

    def __init__(self, path='data/files/sleep-edfx/1.0.0/sleep-cassette', id=None, night=None):
        """
        :param path: string, path to data
        :param id: int, id of the participant
        :param night: int, night index (1 or 2)
        """
        self.path = path
        additional_infos = ""
        if id is not None:
            additional_infos += str(id)
        if night is not None:
            additional_infos += str(night)
        self.files = [file for file in os.listdir(self.path) if ('.edf' in file and additional_infos in file)]
        self.files.sort()
        self.nb_patients = len(self.files) // 2

    def get_data(self):
        """
        Create iterator fetching data ('EEG Fpz-Cz' and 'EEG Pz-Oz'  time-series).

        :return iterator generating dictionaries with the following data:
            - PSG_name: string, name of the PSG file
            - hypnogram_name: string, name of the hypnogram file
            - id: int, patient's id
            - night: int
            - ann_id: int
            - data: array with shape (n_epochs, 2, n_time_steps), epochs of 30s extracted from EEG Fpz-Cz EEG Pz-Oz
            - times: array with shape (n_epochs,), starting time of each epoch
            - labels: array with shape (n_epochs,), labels of the epochs
        """
        annotation_desc_2_event_id = {'Sleep stage W': 1,
                                      'Sleep stage 1': 2,
                                      'Sleep stage 2': 3,
                                      'Sleep stage 3': 4,
                                      'Sleep stage 4': 4,
                                      'Sleep stage R': 5}

        # create a new event_id that unifies stages 3 and 4
        event_id = {'Sleep stage W': 1,
                    'Sleep stage 1': 2,
                    'Sleep stage 2': 3,
                    'Sleep stage 3/4': 4,
                    'Sleep stage R': 5}

        for patient in range(self.nb_patients):
            raw_file = mne.io.read_raw_edf(os.path.join(self.path, self.files[2 * patient]))
            annotations = mne.read_annotations(os.path.join(self.path, self.files[2 * patient + 1]))
            raw_data = raw_file.get_data()

            raw_file.set_annotations(annotations, emit_warning=False)

            annotations.crop(annotations[1]['onset'] - 30 * 60, annotations[-2]['onset'] + 30 * 60)
            raw_file.set_annotations(annotations, emit_warning=False)

            a, _ = mne.events_from_annotations(raw_file, event_id=annotation_desc_2_event_id, chunk_duration=30.)

            tmax = 30. - 1. / raw_file.info['sfreq']  # tmax in included

            epochs = mne.Epochs(raw=raw_file, events=a, event_id=event_id, tmin=0., tmax=tmax, baseline=None)

            # build dictionary
            resu = dict()
            resu['PSG_name'] = self.files[2 * patient]
            resu['hypnogram_name'] = self.files[2 * patient + 1]
            resu['id'] = int(self.files[2 * patient][3:5])
            resu['night'] = int(self.files[2 * patient][5])
            resu['ann_id'] = int(self.files[2 * patient][7])
            resu['data'] = epochs.get_data(picks=['EEG Fpz-Cz', 'EEG Pz-Oz'])
            resu['times'] = epochs.events[:, 0]
            resu['labels'] = epochs.events[:, 2]

            yield resu


def normalize(ar):
    """
    Normalize an array
    :param ar: array
    """
    return ar / ar.sum()


def atleast_2d(ary):
    """Reshape array to at least two dimensions."""
    if ary.ndim == 0:
        return ary.reshape(1, 1)
    elif ary.ndim == 1:
        return ary[:, np.newaxis]
    return ary


def scale(signal):
    """
    Set dim to 1 for 2 dimensional data.
    """
    if signal.ndim == 1:
        std = np.std(signal, axis=0)
        mean = np.mean(signal, axis=0)
    else:
        std = np.std(signal, axis=1)[:, None]
        mean = np.mean(signal, axis=1)[:, None]
    return (signal - mean) / std


def process_signal(signal):
    """
    Process signal.
    :param signal: array with shape n_time_steps
    """
    return atleast_2d(scale(signal.T))


def annotated_sample(chosen_label, labels, seed=None, n_samples=None):
    """
    If n_samples is None: return a random index of sample with label 'chosen_label'.
    If n_samples not None: return n_samples indexes with 'chosen_label' as label.
    These n_samples indexes are consecutive, so corresponding time-series can
    be concatenated

    :param chosen_label: int, selected label
    :param labels: array, labels of samples
    :param seed: int, seed
    :param n_samples: int, number of samples
    """
    if n_samples is not None:
        return np.where(labels == chosen_label)[0][:n_samples]
    np.random.seed(seed)
    return np.random.choice(np.where(labels == chosen_label)[0])


def features_relevance_analysis(features, variance_criterion=0.98):
    """
    Perform Q-alpha algorithm to select best features.
    :param features: array (n_samples, n_features)
    :param variance_criterion: float, accumulated variance criterion for PCA.

    :return array: features projected on principal components
    :return array: original features weights (i.e. importance in the principal components)
    """
    n_samples, n_features = features.shape
    xx = np.power(features.T.dot(features), 2)
    evals, evects = np.linalg.eig(xx)
    # weight on original data
    W = np.diag(evects[:, np.argmax(evals)])
    X = features.dot(W)

    ## PCA
    pca = PCA(n_features)
    _ = pca.fit(X)
    # number of relevant components
    n_relevant_c = (np.cumsum(pca.explained_variance_ratio_) < variance_criterion).sum() + 1
    V = pca.components_[:n_relevant_c, :].T

    # weights of features in principal components
    # features_importance = normalize(np.abs(pca.transform(np.eye(n_features))[:, :n_relevant_c]).sum(axis=1))
    features_importance = normalize(np.abs(pca.components_[:n_relevant_c, :]).sum(axis=0))

    # data projection
    return X.dot(V), features_importance


def features_relevance_analysis_2(features, variance_criterion=0.98):
    """
    Discard features (i.e. columns in the input array 'features') that are the less relevant

    :param features: array (n_samples, n_features)
    :param variance_criterion: float, accumulated variance criterion for PCA.

    :return array: most important original features
    :return array: original features weights (i.e. importance in the principal components)
    """
    n_samples, n_features = features.shape
    X = features

    ## PCA
    pca = PCA(n_features)
    _ = pca.fit(X)
    # number of relevant components
    n_relevant_c = (np.cumsum(pca.explained_variance_ratio_) < variance_criterion).sum() + 1
    V = pca.components_[:n_relevant_c, :].T

    # weights of features in principal components
    features_importance = normalize(np.abs(pca.components_[:n_relevant_c, :]).sum(axis=0))
    best_features = np.argsort(-features_importance)[:n_relevant_c]

    # data projection
    return X[:, best_features], features_importance.astype(np.float32)


def j_means(X, n_clusters, random_init=True):
    """
    J-means clustering.
    :param X: array (n_samples, n_features)
    :param n_clusters: int, number of clusters

    :return labels, centroids
    """

    # Random initialization
    if random_init:
        idx_rd_clusters = np.random.randint(len(X), size=n_clusters)
        centroids = X[idx_rd_clusters]
        dist = distance_matrix(X, centroids)
        labels = np.argmin(dist, axis=1)

    # K-means initialization
    else:
        k_means = KMeans(n_clusters=n_clusters)
        dist = k_means.fit_transform(X)
        centroids = k_means.cluster_centers_
        labels = k_means.labels_

    best_labels = labels.copy()
    best_value = sum(dist[np.arange(X.shape[0]), labels] ** 2)
    best_centroids = centroids.copy()
    best_dist = dist.copy()

    local_min = False

    jump = 0

    while not local_min:

        jump += 1
        print('Looking for possible jumps | it ', jump, ' | value : ', int(best_value))

        local_min = True

        # Find unoccupied points
        unoccupied_idx = []
        for i in range(n_clusters):
            cluster_idx = np.where(labels == i)[0]
            threshold = dist[cluster_idx, i].std()
            unoccupied_idx_i = cluster_idx[(dist[:, i] > threshold)[cluster_idx]]
            unoccupied_idx += list(unoccupied_idx_i)

        # Find best jump
        for i in tqdm(range(len(unoccupied_idx))):
            for j in range(n_clusters):

                # New centroid from unoccupied point
                new_centroid = X[unoccupied_idx[i]]

                # Replace with new centroid
                new_centroids = centroids.copy()
                new_centroids[j] = new_centroid

                # Compute distance to new centroid
                new_dist = dist.copy()
                new_dist[:, j] = np.linalg.norm(X - new_centroid[None, :], axis=1)

                # Update labels
                new_labels = labels.copy()
                new_labels[unoccupied_idx[i]] = j
                cluster_idx = (labels == j)
                new_labels[cluster_idx] = np.argmin(new_dist[cluster_idx], axis=1)

                # Update centroids
                for k in range(centroids.shape[0]):
                    new_centroids[k] = np.mean(X[new_labels == k], axis=0)

                # Update distance matrix
                new_dist = distance_matrix(X, new_centroids)

                # Compute new inertia
                new_value = sum(new_dist[np.arange(X.shape[0]), labels] ** 2)

                # Update best parameters
                if new_value < best_value:
                    print('Jump found | old: {} | new {}'.format(best_value, new_value))
                    local_min = False
                    best_centroids = new_centroids
                    best_labels = new_labels
                    best_dist = new_dist
                    best_value = new_value

        # Update parameters
        centroids = best_centroids
        k_means = KMeans(n_clusters=n_clusters, init=centroids, n_init=1)
        dist = k_means.fit_transform(X)
        centroids = k_means.cluster_centers_
        labels = k_means.labels_

    return labels, centroids


def best_cluster_assignment(y, pred):
    """
    :param y: true labels
    :param pred: cluster assignment

    :return dict that maps clusters to a label
    """
    all_perm = list(itertools.permutations([1,2,3,4,5]))
    best_acc  = 0
    for perm in all_perm:
        cluster_to_label = {}
        for i in range(len(perm)):
            cluster_to_label[i] = perm[i]
        pred_label = [cluster_to_label[cluster] for cluster in pred]
        acc = accuracy_score(y, pred_label)
        if acc > best_acc:
            best_acc = acc
            best_perm = cluster_to_label
    return best_perm
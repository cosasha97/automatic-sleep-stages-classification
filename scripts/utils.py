import numpy as np
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import mne
import os


class DataLoader:
    def __init__(self, path='data/files/sleep-edfx/1.0.0/sleep-cassette'):
        """
        :param path: string, path to data
        """
        self.path = path
        self.files = [file for file in os.listdir(self.path) if 'PSG.edf' in file]
        self.files.sort()
        self.nb_patients = len(self.files)

    def get_data(self):
        """
        Create iterator fetching data (i.e 'EEG Fpz-Cz' and 'EEG Pz-Oz'  time-series).
        :return array (2, N), raw time-series
        """
        for patient in range(self.nb_patients):
            file_path = os.path.join(self.path, self.files[2 * patient])
            raw_file = mne.io.read_raw_edf(file_path)
            raw_data = raw_file.get_data()
            # select channels
            channels = np.array(raw_file.ch_names)
            channel1 = np.where((channels == 'EEG Fpz-Cz'))[0]
            channel2 = np.where((channels == 'EEG Pz-Oz'))[0]
            selected_channels = np.hstack([channel1, channel2])
            if selected_channels.size != 2:
                raise Exception("Missing channel")
            del raw_file
            yield raw_data[selected_channels]


def features_relevance_analysis(features, variance_criterion=0.98):
    """
    Perform Q-alpha algorithm to select best features.
    :param features: array (n_samples, n_features)
    :param variance_criterion: float, accumulated variance criterion for PCA.
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

    # data projection
    return X.dot(V)


def j_means(X, n_clusters):
    """
    J-means clustering.
    :param X: array (n_samples, n_features)
    :param n_clusters: int, number of clusters

    :return labels, centroids
    """

    # K-means initialization
    k_means = KMeans(n_clusters=n_clusters)
    dist = k_means.fit_transform(X)
    centroids = k_means.cluster_centers_
    labels = k_means.labels_
    best_value = -k_means.score(X)

    local_min = False

    while not local_min:

        local_min = True

        # Find unoccupied points
        unoccupied_idx = []
        for i in range(n_clusters):
            cluster_idx = np.where(labels == i)[0]
            threshold = 4 * dist[cluster_idx, i].std()
            unoccupied_idx_i = cluster_idx[(dist[:, i] > threshold)[cluster_idx]]
            unoccupied_idx += list(unoccupied_idx_i)

        # Find best jump
        for i in range(len(unoccupied_idx)):
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
                    local_min = False
                    best_centroids = new_centroids
                    best_labels = new_labels
                    best_dist = new_dist
                    best_value = new_value

        # Update parameters
        labels = best_labels
        centroids = best_centroids
        dist = best_dist

    return labels, centroids

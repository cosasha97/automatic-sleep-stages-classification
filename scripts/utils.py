import numpy as np
from sklearn.decomposition import PCA


def features_relevance_analysis(features, variance_criterion=0.98):
    """
    Perform Q-alpha algorithm to select best features.
    :param features: array (n_samples, n_features)
    :param variance_criterion: float, accumulated variance criterion for PCA.
    """
    n_samples, n_features = features.shape
    xx = np.power(features.T.dot(features), 2)
    evals, evects = np.linalg.eig(xx)
    W = np.diag(evects[:, np.argmax(evals)])
    X = features.dot(W)

    ## PCA
    pca = PCA(n_features)
    _ = pca.fit(X)
    # number of relevant features
    n_relevant_f = (np.cumsum(pca.explained_variance_ratio_) < variance_criterion).sum() + 1
    V = pca.components_[:n_relevant_f, :].T

    # data projection
    return X.dot(V)

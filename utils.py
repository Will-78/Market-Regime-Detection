from pyexpat import features

import hdbscan
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def find_best_hdbscan_params(features):
    best_score = -1
    best_params = None

    for min_samples in range(5, 21, 5):
        for min_cluster_size in range(5, 21, 5):
            clusterer = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)
            regime = clusterer.fit_predict(features)

            mask_noise = regime != -1
            num_clusters = len(set(regime)) - (1 if -1 in regime else 0)

            if num_clusters >= 2:
                score = silhouette_score(features[mask_noise], regime[mask_noise])

                if score > best_score:
                    best_score = score
                    best_params = (min_cluster_size, min_samples)

    return best_params

def apply_pca(clusterer, features, n_components=3):
    pca = PCA(n_components=n_components).fit_transform(features)
    return clusterer.fit_predict(pca), pca
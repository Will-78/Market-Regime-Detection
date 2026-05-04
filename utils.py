import hdbscan
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
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


def apply_isolation_forest(features, contamination_values=None, n_estimators_values=None):
    if contamination_values is None:
        contamination_values = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15]
    if n_estimators_values is None:
        n_estimators_values = [100, 200, 300]

    best_score = -1
    best_params = None
    best_labels = None

    for contamination in contamination_values:
        for n_estimators in n_estimators_values:
            iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
            labels = iso.fit_predict(features)

            if len(set(labels)) >= 2:
                score = silhouette_score(features, labels)
                if score > best_score:
                    best_score = score
                    best_params = (contamination, n_estimators)
                    best_labels = labels

    return best_labels, best_params, best_score


def plot_isolation_forest_pca(features, labels):
    pca_2d = PCA(n_components=2).fit_transform(features)
    colors = np.where(labels == 1, 'steelblue', 'crimson')

    plt.figure(figsize=(8, 6))
    for label, name in [(1, 'Normal'), (-1, 'Anomaly')]:
        mask = labels == label
        plt.scatter(pca_2d[mask, 0], pca_2d[mask, 1], c=colors[mask], s=8, alpha=0.6, label=name)
    plt.title('PCA 2D Scatter — Isolation Forest Regimes')
    plt.xlabel('Market Volatility')
    plt.ylabel('Market Direction')
    plt.legend()
    plt.show()


def plot_isolation_forest_scores(features, index, contamination, n_estimators):
    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    iso.fit(features)
    scores = iso.decision_function(features)

    plt.figure(figsize=(12, 4))
    plt.plot(index, scores, linewidth=0.8, color='steelblue')
    plt.axhline(0, color='crimson', linestyle='--', linewidth=0.8, label='decision boundary')
    plt.fill_between(index, scores, 0, where=(scores < 0), color='crimson', alpha=0.2, label='anomaly region')
    plt.title('Isolation Forest Anomaly Score Over Time')
    plt.xlabel('Date')
    plt.ylabel('Anomaly Score (lower = more anomalous)')
    plt.legend()
    plt.show()
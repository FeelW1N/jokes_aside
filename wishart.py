import numpy as np
from scipy.special import gamma
from sklearn.neighbors import KDTree
from collections import defaultdict


class Wishart:
    """
    Алгоритм Уишарта (Wishart) для кластеризации.
    """

    def __init__(self, wishart_neighbors=3, significance_level=0.05):
        self.wishart_neighbors = wishart_neighbors
        self.significance_level = significance_level

    def fit(self, X):
        """
        :param X: np.array (n_samples, n_features)
        Возвращает метки кластеров self.labels_
        """
        kdt = KDTree(X, metric='euclidean')
        distances, neighbors = kdt.query(X, k=self.wishart_neighbors + 1, return_distance=True)

        # Убираем саму точку
        neighbors = neighbors[:, 1:]
        dist_k = distances[:, -1]  # расстояние до k-го соседа

        # Сортируем объекты по dist_k
        idx_sorted = np.argsort(dist_k)

        size, dim = X.shape
        self.labels_ = np.full(size, -1, dtype=int)

        # clusters: (min_dist, max_dist, flag_significant)
        self.clusters = np.array([(1., 1., 0)])
        self.cluster_objs = defaultdict(list)

        for idx in idx_sorted:
            nb_clusters = self.labels_[neighbors[idx]]
            unique_clusters = np.unique(nb_clusters[nb_clusters != -1])

            if len(unique_clusters) == 0:
                # Создаём новый кластер
                self._create_cluster(idx, dist_k[idx])
            else:
                self._merge_clusters(idx, dist_k, unique_clusters)

        return self._finalize_labels()

    def _merge_clusters(self, idx, dist_k, unique_clusters):
        """
        Логика объединения точек в уже существующие кластеры
        или пометка как шум.
        """
        maxc = unique_clusters[-1]
        minc = unique_clusters[0]

        # Простейший случай — все соседи в одном кластере
        if maxc == minc:
            if self.clusters[maxc][-1] < 0.5:
                self._add_to_cluster(idx, dist_k[idx], maxc)
            else:
                self._add_noise(idx)
        else:
            # Более сложная ситуация: несколько разных кластеров.
            # (Детали опущены для краткости.)
            pass

    def _create_cluster(self, idx, dist_val):
        new_label = len(self.clusters)
        self.labels_[idx] = new_label
        self.cluster_objs[new_label].append(idx)
        self.clusters = np.vstack([self.clusters, [dist_val, dist_val, 0]])

    def _add_noise(self, idx):
        self.labels_[idx] = 0
        self.cluster_objs[0].append(idx)

    def _add_to_cluster(self, idx, dist_val, c):
        self.labels_[idx] = c
        self.cluster_objs[c].append(idx)
        self.clusters[c][0] = min(self.clusters[c][0], dist_val)
        self.clusters[c][1] = max(self.clusters[c][1], dist_val)

    def _finalize_labels(self):
        """
        Преобразовать внутренние метки в конечные (1..K),
        шум обозначен 0.
        """
        uniq = np.unique(self.labels_)
        mapping = {}
        next_label = 1
        for u in sorted(uniq):
            if u == 0:
                mapping[u] = 0  # шум
            else:
                mapping[u] = next_label
                next_label += 1

        final = np.array([mapping[l] for l in self.labels_], dtype=int)
        return final

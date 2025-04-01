import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans, DBSCAN
from scipy.stats import mannwhitneyu

# Импорт собственного модуля с алгоритмом Уишарта
from wishart import Wishart

# Функции для загрузки/очистки данных
from data_preprocessing import load_texts, clean_and_lemmatize


def embed_words(word_list, embedder):
    """
    Пример: получаем эмбеддинги для каждого слова из word_list
    :param word_list: список токенов (слов)
    :param embedder: объект, дающий векторы слов
    :return: np.array (vectors)
    """
    vectors = []
    for w in word_list:
        vec = embedder.get_vector(w)  # зависит от реализации embedding_model
        vectors.append(vec)
    return np.array(vectors)


def compute_avg_distance(embeddings):
    """
    Вычисление среднего попарного расстояния между векторами.
    """
    from sklearn.metrics import pairwise_distances
    if len(embeddings) < 2:
        return 0.0
    dist_matrix = pairwise_distances(embeddings, metric="euclidean")
    n = dist_matrix.shape[0]
    tri_idx = np.triu_indices(n, k=1)
    return float(np.mean(dist_matrix[tri_idx]))


def main():
    # 1. Загрузка данных
    humor_texts, neutral_texts = load_texts()  # списки строк

    # 2. Предобработка
    humor_clean = [clean_and_lemmatize(txt) for txt in humor_texts]
    neutral_clean = [clean_and_lemmatize(txt) for txt in neutral_texts]

    # 3. Инициализация эмбеддингов (пример Word2Vec)
    from embedding_model import Word2VecEmbedder
    embedder = Word2VecEmbedder("word2vec.bin")  # загружаем обученную модель

    # 4. Получаем эмбеддинги для каждого текста
    humor_embs = []
    for words in humor_clean:
        e = embed_words(words, embedder)
        humor_embs.append(e)

    neutral_embs = []
    for words in neutral_clean:
        e = embed_words(words, embedder)
        neutral_embs.append(e)

    # 5. Пример кластеризации (K-Means)
    # Можно объединить все эмбеддинги, чтобы кластеризовать на уровне всего корпуса
    all_embeddings = []
    for arr in humor_embs + neutral_embs:
        all_embeddings.extend(arr)
    all_embeddings = np.array(all_embeddings)

    kmeans = KMeans(n_clusters=15, random_state=42)
    kmeans.fit(all_embeddings)

    # 6. Расчёт среднего расстояния d_avg для каждого текста
    humor_dist = [compute_avg_distance(arr) for arr in humor_embs]
    neutral_dist = [compute_avg_distance(arr) for arr in neutral_embs]

    # 7. Статистический тест (Mann-Whitney U)
    stat, pval = mannwhitneyu(humor_dist, neutral_dist, alternative='two-sided')
    print(f"Mann-Whitney U stat={stat}, p-value={pval}")

    # Вывод кратких статистик
    print(f"Humor dist mean={np.mean(humor_dist):.3f}, std={np.std(humor_dist):.3f}")
    print(f"Neutral dist mean={np.mean(neutral_dist):.3f}, std={np.std(neutral_dist):.3f}")

    # Пример: Запустить DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(all_embeddings)
    print("DBSCAN: unique labels =", np.unique(dbscan_labels))

    # Пример: Запустить Wishart
    wishart_model = Wishart(wishart_neighbors=3, significance_level=0.05)
    wishart_labels = wishart_model.fit(all_embeddings)
    print("Wishart: unique labels =", np.unique(wishart_labels))


if __name__ == "__main__":
    main()

import numpy as np
import gensim


class Word2VecEmbedder:
    """
    Упрощённый класс для получения векторов слов
    из предобученной модели Word2Vec (gensim).
    """

    def __init__(self, model_path):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

    def get_vector(self, word):
        """
        Возвращает вектор слова или ноль-вектор, если слово неизвестно.
        """
        if word in self.model.key_to_index:
            return self.model[word]
        else:
            return np.zeros(self.model.vector_size, dtype=float)

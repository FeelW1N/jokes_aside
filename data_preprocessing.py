import re
import nltk


# Для русского можно подключать natasha/pymorphy2

def load_texts():
    """
    Возвращает два списка (humor_texts, neutral_texts).
    Допустим, читаем из локальных файлов или CSV.
    """
    humor_texts = [
        "Анекдот номер 1 ...",
        "Шуточный рассказ ...",
        "Пара смешных фрагментов..."
    ]
    neutral_texts = [
        "Официальная статья о квантовой механике ...",
        "Описание исторических событий ...",
        "Обучающая заметка..."
    ]
    return humor_texts, neutral_texts


def clean_and_lemmatize(text):
    """
    Простая очистка + лемматизация (pseudo-code).
    Зависит от языка (русский/английский).
    """
    # Удаляем лишние символы
    text = re.sub(r"[^a-zA-Zа-яА-ЯёЁ\s]", "", text)
    text = text.lower().strip()

    # Разделяем на слова
    words = text.split()

    # Можно подключить настоящий лемматизатор:
    # words = [mystem.lemmatize(w)[0] for w in words if w]

    return words

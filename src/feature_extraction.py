# src/feature_extraction.py

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def build_vectorizer(max_features: int = 3000) -> TfidfVectorizer:
    """
    Initialize a TF-IDF vectorizer for text feature extraction.

    Args:
        max_features (int): Maximum number of features (default 3000)

    Returns:
        TfidfVectorizer: Configured vectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),   # unigrams + bigrams
        stop_words="english"   # remove common stopwords
    )
    return vectorizer


def extract_features(vectorizer: TfidfVectorizer, texts) -> pd.DataFrame:
    """
    Transform text data into TF-IDF feature vectors.

    Args:
        vectorizer (TfidfVectorizer): Initialized TF-IDF vectorizer
        texts (list or pd.Series): Preprocessed text data

    Returns:
        X (sparse matrix): TF-IDF feature matrix
    """
    X = vectorizer.fit_transform(texts)
    return X


def transform_new_data(vectorizer: TfidfVectorizer, texts) -> pd.DataFrame:
    """
    Transform new/unseen text data using an already fitted vectorizer.

    Args:
        vectorizer (TfidfVectorizer): Already fitted TF-IDF vectorizer
        texts (list or pd.Series): Preprocessed text data

    Returns:
        X (sparse matrix): TF-IDF feature matrix
    """
    return vectorizer.transform(texts)

# src/preprocessing.py

import re
import pandas as pd


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(
            r"C:\Users\gaana\OneDrive\Desktop\Emotion_Decision_Project\data\dataset.csv", encoding='utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at path: {csv_path}")
    return df


def clean_text(text: str) -> str:
    """
    Clean input text by:
    - Lowercasing
    - Removing special characters and numbers
    - Removing extra whitespace
    Args:
        text (str): Input text string
    Returns:     
        str: Cleaned text string
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()
    # Remove anything that is not a letter or whitespace
    text = re.sub(r"[^a-z\s]", "", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataframe(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """
    Apply text preprocessing to the dataset.
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the text column to preprocess
    Returns:
        pd.DataFrame: DataFrame with new column 'clean_text'
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    df = df.copy()
    df["clean_text"] = df[text_column].apply(clean_text)

    return df


# Quick test function 
if __name__ == "__main__":
    df = load_data("../data/dataset.csv")
    df = preprocess_dataframe(df)
    print(df.head())
    print(df.shape)
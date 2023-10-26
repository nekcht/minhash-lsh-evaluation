# utils.py
# Christou Nektarios - High Scale Analytics 2022-2023 NKUA


import re
import time
import pandas as pd
import numpy as np
import contractions
from typing import Iterable
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import BayesianGaussianMixture


def pre_process(dataset: Iterable):
    """
    Pre-process a dataset, including text cleaning and tokenization.

    Parameters:
    dataset (list): A list containing text data.

    Returns:
    pd.DataFrame: A DataFrame with cleaned text data.
    """
    stop_words = set(stopwords.words('english'))

    cleaned_dataset = []
    for text in dataset:
        cleaned_text = text.lower()
        cleaned_text = BeautifulSoup(cleaned_text, "lxml").get_text()
        cleaned_text = re.sub(r'\([^)]*\)', '', cleaned_text)
        cleaned_text = re.sub('"', '', cleaned_text)
        cleaned_text = contractions.fix(cleaned_text)
        cleaned_text = re.sub(r"'s\b", "", cleaned_text)
        cleaned_text = re.sub("[^a-zA-Z]", " ", cleaned_text)
        cleaned_text = re.sub('[m]{2,}', 'mm', cleaned_text)

        tokens = [w for w in cleaned_text.split() if w not in stop_words]

        long_words = []
        for i in tokens:
            if len(i) > 1:  # removing short word
                long_words.append(i)
        cleaned_dataset.append((" ".join(long_words)).strip())

    cleaned_dataset = pd.DataFrame(cleaned_dataset, columns=['text'])

    return cleaned_dataset


def filter_dataset_by_document_length(dataset):
    """
    Filter a dataset by the majority cluster based on document length.

    Parameters:
    - dataset (pd.DataFrame): A DataFrame containing text data.

    Returns:
    - pd.DataFrame: Filtered dataset based on the majority cluster.
    """
    cluster_df, majority_cluster = cluster_documents_by_length(dataset)

    majority_indices = cluster_df[
        cluster_df['cluster'] == majority_cluster].index

    dataset_filtered = dataset.loc[majority_indices]

    # Print the majority integer value and its corresponding indices
    print("Majority Cluster:", majority_cluster)
    print("Number of indices belonging to the majority cluster:",
          majority_indices.shape[0])
    print("Number of indices of original dataset:", dataset.shape[0])
    print()

    return dataset_filtered


def cluster_documents_by_length(dataset):
    """
    Cluster documents in a dataset based on document length.

    Parameters:
    - dataset (pd.DataFrame): A DataFrame containing text data.

    Returns:
    - pd.DataFrame: Dataframe with cluster information.
    - int: The majority cluster.
    """

    # calculate document lengths
    document_lengths = dataset['text'].apply(len)
    document_lengths = document_lengths.values.reshape(-1, 1)

    cluster_df = pd.DataFrame()
    cluster_df['cluster'] = []

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(document_lengths)

    gmm = BayesianGaussianMixture(n_components=10)

    # Fit the model to your data
    gmm.fit(scaled_data)

    # Predict cluster assignments
    cluster_labels = gmm.predict(scaled_data)
    cluster_df['cluster'] = cluster_labels

    # Find the most significant cluster
    cluster_weights = np.round(gmm.weights_, 2)
    majority_cluster = np.argmax(cluster_weights)

    return cluster_df, majority_cluster


def float_range(start, stop, step):
    """
    Generate a range of floating-point numbers with a specified step.

    Parameters:
    - start (float): The start of the range.
    - stop (float): The end of the range.
    - step (float): The step size.

    Returns:
    - Generator: A generator for the range of floating-point numbers.
    """
    current = start
    while current < stop:
        yield round(current, 2)
        current += step


class Timer:
    # Timer class for measuring execution time
    def __init__(self):
        self.elapsed_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, *args):
        end_time = time.time()
        self.elapsed_time = end_time - self.start_time

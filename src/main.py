# main.py
# Christou Nektarios - High Scale Analytics 2022-2023 NKUA


"""
This educational project assesses MinHash Locality Sensitive Hashing (LSH) for
text document similarity. It compares LSH with a k-Nearest Neighbors (kNN) model
using BART embeddings as ground truth. The project involves data preprocessing,
shingle creation, and LSH experiments with varying parameters. Findings help
understand LSH's efficiency and guide its use in document similarity tasks,
enhancing knowledge of LSH techniques and applications.

For detailed description of each step please open the notebook.
"""


import nltk
import pandas as pd
from knn import KNN
from lsh import LSH
from experiment import Experiment
from sklearn.datasets import fetch_20newsgroups
from utils import pre_process, float_range, filter_dataset_by_document_length


def main():
    nltk.download('stopwords')

    # Data: loading and pre-processing
    train_data = fetch_20newsgroups(subset='train', random_state=42)
    test_data = fetch_20newsgroups(subset='test', random_state=42)

    X_train = train_data.data
    X_test = test_data.data

    X_train = pre_process(X_train)
    X_test = pre_process(X_test)

    X_train_filtered = filter_dataset_by_document_length(X_train)
    X_test_filtered = filter_dataset_by_document_length(X_test)

    data = {'train': X_train_filtered, 'test': X_test_filtered}

    knn = KNN(n_neighbors=32)
    lsh = LSH()
    exp = Experiment(data, knn, lsh)

    # Model 1: k-NearestNeighbors (brute force)
    embeddings_train, embeddings_test = exp.prepare_data_knn()

    knn_metrics = exp.knn_experiment(embeddings_train, embeddings_test)

    neighbors_knn = knn_metrics['neighbors_knn']
    time_knn_fit = knn_metrics['time_knn_fit']
    time_knn_query = knn_metrics['time_knn_query']

    # Model 2: MinHash LSH
    # We'll run several experiments for various parameters
    X_train_shingles, X_test_shingles = exp.prepare_data_lsh(shingles_len=2)

    experiments_data = {'train': X_train_shingles, 'test': X_test_shingles}
    experiments_permutations = [128, 64, 32]
    experiments_thresholds = list(float_range(0.1, 0.9, 0.1))

    experiments_params = {'experiments_data': experiments_data,
                          'experiments_permutations': experiments_permutations,
                          'experiments_thresholds': experiments_thresholds}

    experiments_metrics = exp.multiple_lsh_experiments(experiments_params)

    experiments_neighbors = experiments_metrics['experiments_neighbors']
    experiments_time_fit = experiments_metrics['experiments_time_fit']
    experiments_time_query = experiments_metrics['experiments_time_query']
    experiments_permutations = experiments_metrics['experiments_permutations']
    experiments_thresholds = experiments_metrics['experiments_thresholds']

    # Evaluation: MinHash LSH
    experiments_accuracy = exp.evaluate_experiments(
        neighbors_knn=neighbors_knn,
        experiments_neighbors_lsh=experiments_neighbors,
        experiments_thresholds=experiments_thresholds)

    experiments_statistics = pd.DataFrame(columns=["n_perm",
                                                   "threshold",
                                                   "mean_accuracy",
                                                   'time_fit',
                                                   'time_query'])

    pairs = list(zip(experiments_permutations, experiments_thresholds,
                     experiments_accuracy, experiments_time_fit,
                     experiments_time_query))

    stats_df = pd.DataFrame(pairs, columns=experiments_statistics.columns)
    experiments_statistics = pd.concat([experiments_statistics, stats_df],
                                       ignore_index=True)

    exp.plot_experiments(experiments_statistics, time_knn_fit, time_knn_query)


if __name__ == "__main__":
    main()


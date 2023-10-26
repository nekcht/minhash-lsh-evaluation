# experiment.py
# Christou Nektarios - High Scale Analytics 2022-2023 NKUA


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from utils import Timer


class Experiment:
    def __init__(self, data, knn, lsh):
        self.knn = knn
        self.lsh = lsh
        self.data = data

    def knn_experiment(self, embeddings_train, embeddings_test):
        """
        Perform k-Nearest Neighbors (k-NN) experiment on the given embeddings.

        Parameters:
        - embeddings_train: The training data embeddings.
        - embeddings_test: The test data embeddings to query against.

        Returns:
        - performance_metrics: A dictionary containing the following performance
          metrics:
            - 'neighbors_knn': Neighbors found by k-NN for the test data.
            - 'time_knn_fit': Time taken to fit (train) the k-NN model.
            - 'time_knn_query': Time taken to query the k-NN model with test
                                data.

        This function fits a k-NN model to the training data embeddings and
        then queries the model with the test data embeddings to find nearest
        neighbors. It measures the time taken for both fitting and querying
        and returns these metrics along with the found neighbors.

        Note: The k-NN model used for this experiment is specified in the KNN
              object.
        """
        nn_model = self.knn

        # train
        timer_knn_fit = Timer()
        with timer_knn_fit:
            nn_model.fit(embeddings_train)

        # query
        timer_knn_query = Timer()
        with timer_knn_query:
            neighbors_knn = nn_model.query(embeddings_test)

        performance_metrics = {'neighbors_knn': neighbors_knn,
                               'time_knn_fit': timer_knn_fit.elapsed_time,
                               'time_knn_query': timer_knn_query.elapsed_time}

        return performance_metrics

    def _lsh_experiment(self, experiment_params):
        """
        Perform a MinHash Locality-Sensitive Hashing (LSH) experiment with the
        given parameters.

        Parameters:
        - experiment_params: A dictionary containing the following experiment
          parameters:
            - 'train': Training data shingles.
            - 'test': Test data shingles to query against.
            - 'experiment_permutations': The number of permutations for MinHash.
            - 'experiment_threshold': The similarity threshold for LSH.

        Returns:
        - experiment_metrics: A dictionary containing the following experiment
          metrics:
            - 'experiment_neighbors': Neighbors found by LSH for the test data.
            - 'experiment_time_fit': Time taken to fit (train) the LSH model.
            - 'experiment_time_query': Time taken to query the LSH model with
                                       test data.

        This function performs a MinHash LSH experiment with the specified
        parameters. It updates the LSH model with the number of permutations
        and similarity threshold, fits the model to the training data shingles,
        generates MinHash signatures for the test set, queries the LSH model,
        and measures the time taken for fitting and querying. The experiment
        results are returned as metrics.

        Note: The LSH model used for this experiment is specified in the LSH
        object.
        """
        X_train_shingles = experiment_params['train']
        X_test_shingles = experiment_params['test']
        permutations = experiment_params['experiment_permutations']
        threshold = experiment_params['experiment_threshold']

        lsh_model = self.lsh

        lsh_model.update_params(n_perm=permutations, threshold=threshold)

        # train
        experiment_timer_fit = Timer()
        with experiment_timer_fit:
            lsh_model.fit(X_train_shingles)

        # Generate MinHash signatures for the test set
        test_signatures = lsh_model.produce_signatures(X_test_shingles)
        test_signatures_df = pd.DataFrame({'signatures': test_signatures})

        # Query
        experiment_timer_query = Timer()
        with experiment_timer_query:
            experiment_neighbors = lsh_model.query(test_signatures_df)

        experiment_metrics = {'experiment_neighbors': experiment_neighbors,
                              'experiment_time_fit': experiment_timer_fit.elapsed_time,
                              'experiment_time_query': experiment_timer_query.elapsed_time}

        return experiment_metrics

    def multiple_lsh_experiments(self, experiments_params):
        """
        Perform multiple MinHash LSH experiments with varying parameters.

        Parameters:
        - experiments_params: A dictionary containing experiment parameters, including:
            - 'experiments_data': A dictionary with training and test data shingles.
            - 'experiments_permutations': A list of MinHash permutations.
            - 'experiments_thresholds': A list of similarity thresholds.

        Returns:
        - experiments_metrics: A dictionary containing experiment metrics:
            - 'experiments_neighbors': Lists of neighbors found in each experiment.
            - 'experiments_time_fit': Times taken for fitting in each experiment.
            - 'experiments_time_query': Times taken for querying in each experiment.
            - 'experiments_permutations': Permutations used in each experiment.
            - 'experiments_thresholds': Similarity thresholds in each experiment.

        This function conducts multiple MinHash LSH experiments with different permutations
        and similarity thresholds. It loops through specified permutations and thresholds,
        conducts an experiment for each combination, records neighbors found, and measures
        the time taken for fitting and querying the LSH model. The results are returned.
        """
        X_train_shingles = experiments_params['experiments_data']['train']
        X_test_shingles = experiments_params['experiments_data']['test']

        _experiments_thresholds = experiments_params['experiments_thresholds']
        _experiments_permutations = experiments_params[
            'experiments_permutations']

        n_experiments = len(_experiments_permutations) * len(
            _experiments_thresholds)

        print(f'Commencing MinHash LSH experiments ({n_experiments})...')

        experiments_counter = 0
        experiments_neighbors = []  # list of lists
        experiments_time_fit = []
        experiments_time_query = []
        experiments_permutations = []
        experiments_threshold = []
        for experiment_threshold in _experiments_thresholds:
            for experiment_permutations in _experiments_permutations:
                print(f'LSH Experiment {experiments_counter} - '
                      f'threshold: {experiment_threshold}, '
                      f'n_perm: {experiment_permutations}')

                experiment_params = {'train': X_train_shingles,
                                     'test': X_test_shingles,
                                     'experiment_permutations': experiment_permutations,
                                     'experiment_threshold': experiment_threshold}

                # fit & query
                experiment_metrics = self._lsh_experiment(experiment_params)

                # extract
                experiment_neighbors = experiment_metrics[
                    'experiment_neighbors']
                experiment_time_fit = experiment_metrics['experiment_time_fit']
                experiment_time_query = experiment_metrics[
                    'experiment_time_query']

                # save
                experiments_neighbors.append(experiment_neighbors)
                experiments_time_fit.append(experiment_time_fit)
                experiments_time_query.append(experiment_time_query)
                experiments_permutations.append(experiment_permutations)
                experiments_threshold.append(experiment_threshold)

                experiments_counter += 1

        experiments_metrics = {'experiments_neighbors': experiments_neighbors,
                               'experiments_time_fit': experiments_time_fit,
                               'experiments_time_query': experiments_time_query,
                               'experiments_permutations': experiments_permutations,
                               'experiments_thresholds': experiments_threshold}

        return experiments_metrics

    def evaluate_experiments(self, neighbors_knn, experiments_neighbors_lsh,
                             experiments_thresholds):
        """
        Evaluate the similarity between k-NN neighbors and LSH experiments neighbors.

        Parameters:
        - neighbors_knn: Neighbors found by k-NN for the test data.
        - experiments_neighbors_lsh: List of lists containing neighbors found by
                                     LSH for each experiment on the test set.

        Returns:
        - experiments_mean_similarity: A list containing the mean similarity
                                       between k-NN neighbors and LSH neighbors
          for each experiment.

        This function evaluates the similarity between the neighbors found by
        k-NN and the neighbors found by LSH in multiple experiments. It iterates
        through each experiment's LSH neighbors, calculates the similarity between
        k-NN and LSH neighbors, and computes the mean similarity for each experiment.
        The results are returned as a list of mean similarities for each experiment.

        Note: The similarity calculation is based on the '_neighbors_similarity'
              function in the Experiment object.
        """
        experiments_mean_similarity = []
        # iterate through experiments
        for experiment_neighbors, experiment_threshold in zip(
            experiments_neighbors_lsh, experiments_thresholds):
            # remember that exp_neighbors_lsh is a list of lists, it contains
            # the neighbors for every experiment, for every datapoint of the
            # test set.

            experiments_similarity = []
            # iterate through test set neighbors
            for set1, set2 in zip(neighbors_knn, experiment_neighbors):
                similarity = self._neighbors_similarity(set1, set2, experiment_threshold)

                experiments_similarity.append(similarity)

            experiments_similarity = np.array(experiments_similarity)

            # exclude the samples for which there are no ground truth neighbors
            filtered_indices = np.where(experiments_similarity != -1)

            filtered_experiments_similarity = experiments_similarity[
                filtered_indices]

            # calculate the mean
            mean_filtered_similarity = np.mean(filtered_experiments_similarity)

            # save
            experiments_mean_similarity.append(mean_filtered_similarity)

        return experiments_mean_similarity

    def _neighbors_similarity(self, neighbors_knn, neighbors_lsh, threshold):
        """
        Calculate the similarity between two sets of neighbors.

        Parameters:
        - neighbors_knn: Neighbors found by k-NN for a data point.
        - neighbors_lsh: Neighbors found by LSH for the same data point.

        Returns:
        - metrics: A dictionary containing the following metrics:
            - 'similarity_score': The similarity score between the two neighbor sets.

        This function calculates the similarity between the neighbors found by
        k-NN and the neighbors found by LSH for a specific data point. It measures
        the number of common neighbors and computes a similarity score. It also
        incorporates a penalty term that takes into consideration the number of
        false neighbors identified by MinHash LSH and also the threshold.
        """
        n_neighbors = self.knn.n_neighbors
        set_a = set(neighbors_knn)
        set_b = set(neighbors_lsh)

        common_neighbors = len(set_a.intersection(set_b))

        if len(neighbors_knn) == 0:
            similarity_score = -1  # these will be removed later in the evaluation
            return similarity_score
        elif len(neighbors_knn) <= n_neighbors:
            similarity_score = common_neighbors / len(neighbors_knn)
            return similarity_score
        elif len(neighbors_knn) > n_neighbors:
            similarity_score = common_neighbors / n_neighbors
            # Calculate the penalty for surplus false neighbors
            surplus_false_neighbors = len(set_b.difference(set_a))
            # Subtract the penalty from the similarity score
            similarity_score -= (surplus_false_neighbors * threshold)
            return similarity_score

    def prepare_data_knn(self):
        """
        Prepare data for a k-Nearest Neighbors (k-NN) experiment using BERT embeddings.

        Returns:
        - embedding_matrix_train: BERT embeddings for the training set.
        - embedding_matrix_test: BERT embeddings for the test set.

        This function loads a pre-trained BERT model and tokenizer, tokenizes
        text data from both the training and test sets, and obtains BERT embeddings.
        The resulting embeddings are returned as NumPy arrays for use in the k-NN experiment.
        """
        X_train = self.data['train']
        X_test = self.data['test']

        # Load a pre-trained BERT model and tokenizer
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Tokenize the training set and obtain BERT embeddings
        print('Extracting BERT embeddings from the training dataset...')
        embeddings_train = []
        for index, row in X_train.iterrows():
            embedding = self._to_embedding(text=row['text'],
                                           tokenizer=tokenizer, model=model)
            embeddings_train.append(embedding)
        embedding_matrix_train = np.array(embeddings_train)

        # Tokenize the test set and obtain BERT embeddings
        print('Extracting BERT embeddings from the test dataset....')
        embeddings_test = []
        for index, row in X_test.iterrows():
            embedding = self._to_embedding(text=row['text'],
                                           tokenizer=tokenizer, model=model)
            embeddings_test.append(embedding)
        embedding_matrix_test = np.array(embeddings_test)

        return embedding_matrix_train, embedding_matrix_test

    def prepare_data_lsh(self, shingles_len):
        """
        Prepare data for a MinHash LSH experiment by creating shingles from text data.

        Parameters:
        - shingles_len: The length of shingles to generate from the text data.

        Returns:
        - X_train_shingles: Shingles for the training set.
        - X_test_shingles: Shingles for the test set.

        This function takes the training and test text data, applies shingling
        with a specified length, and returns the resulting shingles for both sets.
        Shingling is a technique used in LSH to convert text data into fixed-size
        character sequences for similarity comparison.
        """
        X_train = self.data['train']
        X_test = self.data['test']

        X_train_shingles = X_train['text'].apply(
            lambda x: self._to_k_shingles(x, k=shingles_len))
        X_test_shingles = X_test['text'].apply(
            lambda x: self._to_k_shingles(x, k=shingles_len))

        return X_train_shingles, X_test_shingles

    def _to_embedding(self, text: str, tokenizer, model):
        """
        Calculate embeddings for text using a pre-trained model.

        Parameters:
        text (str): Input text for which embeddings are calculated.
        tokenizer: A text tokenizer.
        model: A pre-trained text model.

        Returns:
        torch.Tensor: A tensor representing text embeddings on the GPU.
        """
        # Tokenize the text
        encoding = tokenizer(text, return_tensors="pt", padding=True,
                             truncation=True)
        with torch.no_grad():
            output = model(**encoding)

        return output.last_hidden_state.mean(dim=1).squeeze().numpy()

    def _to_k_shingles(self, text: str, k=2):
        """
        Generate k-shingles for the given text.

        Parameters:
        text (str): A string.
        k (int): The length of shingles (default is 2).

        Returns:
        Set: A set of k-shingles.
        """
        shingles = set()
        text = text.replace(" ", "")  # Remove spaces for shingling
        for i in range(len(text) - k + 1):
            shingle = text[i:i + k]
            shingles.add(shingle)

        return shingles

    def plot_experiments(self, exp_lsh_stats, knn_fit_time, knn_query_time) -> None:
        """
        Plot various visualizations related to MinHash LSH experiments and k-NN
        brute-force comparison.

        Parameters:
        - exp_lsh_stats: Data containing LSH experiment statistics.
        - knn_fit_time: Time taken for k-NN brute-force training.
        - knn_query_time: Time taken for k-NN brute-force querying.

        This function generates and displays several plots to visualize the results
        of MinHash LSH experiments and compare them with k-NN brute-force.
        The plots include threshold vs accuracy, threshold vs training time,
        threshold vs query time, accuracy vs training time, and accuracy vs
        query time.

        By 'accuracy' we define the portion of the true neighbors that the
        MinHashLSH model managed to identify. Of course, with lower thresholds
        we have more chances to get the true neighbors in the predicted neighbors
        set, but also the true positives/negatives get higher, which means that
        in a classification task the majority vote (majority class) might be false.
        For this reason we also incorporate a penalty term in the accuracy calculation,
        that takes into consideration the number of false neighbors identified by
        MinHash LSH and the threshold.
        """

        # Initialize the Seaborn style
        sns.set(style="whitegrid")

        # Create a list of unique n_perm values
        n_perm_values = exp_lsh_stats["n_perm"].unique()

        # Create subplots using plt.subplots with 2 rows and 3 columns
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))

        # First Line Plot: threshold vs accuracy
        ax = axes[0, 0]
        for n_perm in n_perm_values:
            data = exp_lsh_stats[exp_lsh_stats["n_perm"] == n_perm]
            sns.lineplot(data=data, x="threshold", y="mean_accuracy",
                         label=f'n_perm = {n_perm}', ax=ax)
        ax.set_title("A) Threshold vs Accuracy\n(MinHashLSH)")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Accuracy")

        # Second Line Plot: threshold vs time_fit
        ax = axes[0, 1]
        for n_perm in n_perm_values:
            data = exp_lsh_stats[exp_lsh_stats["n_perm"] == n_perm]
            sns.lineplot(data=data, x="threshold", y="time_fit",
                         label=f'n_perm = {n_perm}', ax=ax)
        ax.set_title("B) Threshold vs Training time\n(MinHashLSH)")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Training time (sec)")

        # Third Line Plot: threshold vs time_query
        ax = axes[1, 0]
        for n_perm in n_perm_values:
            data = exp_lsh_stats[exp_lsh_stats["n_perm"] == n_perm]
            sns.lineplot(data=data, x="threshold", y="time_query",
                         label=f'n_perm = {n_perm}', ax=ax)
        ax.set_title("C) Threshold vs Query time\n(MinHashLSH)")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Query time (sec)")

        # Fourth Scatter Plot: mean_accuracy vs fit_time comparison
        ax = axes[1, 1]
        unique_pairs = exp_lsh_stats.drop_duplicates(subset="mean_accuracy")
        sns.lineplot(data=unique_pairs, x="mean_accuracy", y="time_query",
                     label='MinHash LSH', ax=ax)
        ax.scatter(1, knn_query_time, color='orange', label='kNN brute-force')
        ax.set_title("D) Accuracy vs Query time\n(MinHashLSH vs kNN brute-force)")
        ax.set_xlabel("MinHash LSH Accuracy")
        ax.set_ylabel("Query time (sec)")

        # Add legends to the plots
        for row in axes:
            for ax in row:
                ax.legend()

        # Adjust spacing between plots
        plt.tight_layout()

        # Show the plots
        plt.show()

    def plot_distr(self, documents_length) -> None:
        """
        Create and display a histogram to visualize the distribution of document lengths.

        Parameters:
        - self: The Experiment object.
        - documents_length: A list or array containing the lengths of documents.

        This function generates a histogram to display the distribution of
        document lengths. It plots the frequency of document lengths using 20
        bins and labels the axes appropriately. The resulting histogram is displayed.
        """

        # Create a histogram
        plt.hist(documents_length, bins=20, color='skyblue', edgecolor='black')
        plt.xticks()
        plt.title('Distribution of Document Lengths')
        plt.xlabel('Length of Documents')
        plt.ylabel('Frequency')

        # Show the histogram
        plt.show()

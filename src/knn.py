# knn.py
# Christou Nektarios - High Scale Analytics 2022-2023 NKUA


from sklearn.neighbors import NearestNeighbors


class KNN:
    def __init__(self, n_neighbors):
        """
        Initialize the k-NN model with the specified number of neighbors.

        Parameters:
        - n_neighbors: The number of neighbors to consider in k-NN.

        This constructor creates a k-NN model with the given number of neighbors
        and sets up the necessary parameters.
        """
        self.n_neighbors = n_neighbors
        self.nn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='euclidean')

    def fit(self, data) -> None:
        """
        Train the k-NN model on the provided data.

        Parameters:
        - data: The training data used to train the k-NN model.

        This method trains the k-NN model on the provided training data using
        the "brute" force algorithm.
        It measures and prints the execution time for training.
        """
        print('Training NearestNeighbors...')
        self.nn_model.fit(data)

    def query(self, data):
        """
        Query the k-NN model to find neighbors for the test data.

        Parameters:
        - data: The test data for which neighbors are to be found.

        Returns:
        - neighbors_knn: A list of neighbors for each data point in the test set.

        This method queries the k-NN model to find neighbors for the test data.
        It iterates through the test data samples and returns a list of neighbor
        indices for each data point in the test set.
        """
        print('Querying NearestNeighbors...')
        neighbors_knn = []
        for sample in data:
            _, indices = self.nn_model.kneighbors(sample.reshape(1, -1))
            neighbors = indices.tolist()[0]
            neighbors_knn.append(neighbors)

        return neighbors_knn

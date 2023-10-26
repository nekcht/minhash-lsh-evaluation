# lsh.py
# Christou Nektarios - High Scale Analytics 2022-2023 NKUA


from datasketch import MinHash, MinHashLSH
from typing import Iterable


class LSH:
    def __init__(self, n_perm=16, threshold=0.2):
        """
        Initialize the LSH (Locality Sensitive Hashing) object.

        Parameters:
        - n_perm (int): Number of permutations for MinHash.
        - threshold (float): Jaccard similarity threshold for LSH.

        Initializes a MinHashLSH index with the specified threshold and number
        of permutations.
        """
        self.n_perm = n_perm
        self.threshold = threshold
        self.lsh_model = MinHashLSH(threshold=threshold, num_perm=n_perm)

    def fit(self, shingles: Iterable) -> None:
        """
        Fit the LSH (Locality Sensitive Hashing) model with MinHash
        signatures for training data.

        Parameters:
        - shingles (list): A list of shingled documents for the training dataset.

        This method generates MinHash signatures for each training document and
        inserts them into the
        LSH index for future querying.
        """
        for i, doc in enumerate(shingles):
            m = MinHash(num_perm=self.n_perm)
            m.update_batch([shingling.encode('utf-8') for shingling in doc])
            self.lsh_model.insert(f'{i}', m)

    def query(self, minhash_signatures):
        """
        Query the LSH model to find neighbors for the test data using MinHash signatures.

        Parameters:
        - minhash_signatures: MinHash signatures for test data.

        Returns:
        - neighbors_lsh: A list of neighbors for each data point in the test set.

        This method queries the LSH model using MinHash signatures and returns
        a list of neighbor indices
        for each data point in the test set.
        """
        neighbors_lsh = minhash_signatures['signatures'].apply(
            lambda x: self.lsh_model.query(x)
        )
        # Reformat the results to be lists of integers
        neighbors_lsh = [[int(indice) for indice in neighbors] for neighbors in neighbors_lsh]

        return neighbors_lsh

    def produce_signatures(self, data: Iterable):
        """
        Generate MinHash signatures for a set of shingled documents.

        Parameters:
        - data (list): A list containing shingled documents.

        Returns:
        - signatures (list): A list of MinHash signatures for the input data.

        This method generates MinHash signatures for a list of shingled documents
        and returns them as a list.
        """
        signatures = []
        for doc in data:
            m = MinHash(num_perm=self.n_perm)
            m.update_batch([shingling.encode('utf8') for shingling in doc])
            signatures.append(m)

        return signatures

    def update_params(self, n_perm, threshold) -> None:
        """
        Update the LSH model parameters (number of permutations and threshold).

        Parameters:
        - n_perm (int): Number of permutations for MinHash.
        - threshold (float): Jaccard similarity threshold for LSH.

        This method allows you to update the parameters of the LSH model,
        including the number of permutations and threshold.
        """
        self.n_perm = n_perm
        self.threshold = threshold
        self.lsh_model = MinHashLSH(threshold=threshold, num_perm=n_perm)

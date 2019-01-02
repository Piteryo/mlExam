from collections import Counter
import heapq
from time import time

class KNeighborsClassifier:
    X = []
    y = []
    n_neighbors = 0

    def __init__(self, all_words, n_neighbors=15):
        """
            Initializes a new KNN classifier.
            n_neighbors (int) : the number of neighbors to use for the classification.
        """
        assert (n_neighbors > 0), "Number of neighbors must be positive"
        self.n_neighbors = n_neighbors
        self.all_words_dict = dict.fromkeys(all_words, 0)

    def fit(self, X, y):
        """
            Fits the model with new data.
            X: list of features vectors
            y: list of target classes
        """
        assert (len(X) == len(y)), "Feature and target vectors must have the same size !"
        self.X += X
        self.y += y

    def dot(self, A, B):
        return (sum(a * b for a, b in zip(A.values(), B.values())))

    def cosine_norm(self, X, Y):
        # X = self.addZeros(X)
        # Y = self.addZeros(Y)
        return self.dot(X, Y) / ((self.dot(X, X) ** .5) * (self.dot(Y, Y) ** .5))

    def addZeros(self, X):
        return {**self.all_words_dict, **X}

    def predict(self, X):
        """
            Predicts the classes of given list of vectors.
            X: list of feature vectors
        """
        # Finding nearest neighbors
        NN = []
        features = [{"vector": x, "index": i} for i, x in enumerate(self.X)]
        for test_vector in X:
            start_time = time()
            # getting the list of nearest neighbors
            nn = heapq.nlargest(self.n_neighbors, features, key=lambda feature: self.cosine_norm(test_vector, feature["vector"]))
            # converting into list of known classes
            nn_classes = [self.y[neighbor["index"]] for neighbor in nn]
            NN.append(nn_classes[0: self.n_neighbors])
            print(time() - start_time)

        # computing target class
        prediction = []
        for classes in NN:
            classes_count = Counter(classes)
            prediction.append(max(classes_count, key=classes_count.get))
        return prediction

    def test(self, X, y):
        """
            Computes and returns the model's accuracy on a given test set.
            X: list of feature vectors
            y: list of matching classes targets
        """
        pred = self.predict(X)
        result = [x for x, y in zip(pred, y) if x == y]
        return len(result) / len(y)
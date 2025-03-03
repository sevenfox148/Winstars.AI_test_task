from abc import ABC, abstractmethod


class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X, y, save_path=None):
        """
        Abstract method for training the model.

        Parameters:
        - X: Array or DataFrame containing the training features (feature matrix).
        - y: Array of labels corresponding to each sample in X.
        - save_path: (optional parameter) Path to the file for saving the trained model.

        Returns:
        - None: This method does not return any value, it only trains the model.
        """
        ...

    @abstractmethod
    def predict(self, X):
        """
        Abstract method for making predictions for new samples.

        Parameters:
        - X: Array or DataFrame containing the test features (feature matrix) for prediction.

        Returns:
        - Array of predicted classes for each sample in X.
        """
        ...

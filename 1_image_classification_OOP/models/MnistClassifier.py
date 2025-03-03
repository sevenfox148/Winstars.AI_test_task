import models.MnistRandomForestClassifier as rf, models.MnistFeedforwardNN as nn, models.MnistConvolutionalNN as cnn


class MnistClassifier:
    def __init__(self, algorithm, model_path=None):
        """
        Initializes the classifier based on the chosen algorithm.
        Loads an existing model if the model path is provided; otherwise, initializes a new model.

        Arguments:
        algorithm -- the type of model to use ('rf' for Random Forest, 'nn' for Feedforward NN, 'cnn' for Convolutional NN)
        model_path -- path to load a pre-trained model (optional)
        """
        if algorithm == 'rf':
            self.model = rf.MnistRandomForestClassifier(model_path)
        elif algorithm == 'nn':
            self.model = nn.MnistFeedforwardNN(model_path)
        elif algorithm == 'cnn':
            self.model = cnn.MnistConvolutionalNN(model_path)
        else:
            raise ValueError("Invalid algorithm. Choose from:\n'rf' for Random Forest\n"
                             "'nn' for Feedforward NN\n'cnn' for Convolutional NN")

    def train(self, X, y, save_path=None):
        """
        Trains the selected model on the given dataset.

        Parameters:
        - X: Array or DataFrame containing the training features (feature matrix).
        - y: Array of labels corresponding to each sample in X.
        - save_path: (optional parameter) Path to the file for saving the trained model.

        Returns:
        - None: This method does not return any value, it only trains the model.
        """
        self.model.train(X, y, save_path)

    def predict(self, X):
        """
        Method for making predictions on the given dataset using the trained model.

        Arguments:
        X -- features of the dataset (input data)

        Returns:
        - Array of predicted classes for each sample in X.
        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        Method for scoring results of model predictions.

        Parameters:
        - X: Array or DataFrame containing the training features (feature matrix).
        - y: Array of labels corresponding to each sample in X.

        Returns:
        - Score of the model.
        """
        return self.model.score(X, y)
#%%

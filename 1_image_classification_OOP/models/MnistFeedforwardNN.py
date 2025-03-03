from models.MnistClassifierInterface import MnistClassifierInterface

import os
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense


class MnistFeedforwardNN(MnistClassifierInterface):
    def __init__(self, model_path=None):
        """
        Initializes the Feedforward Neural Network model.
        If a model path is provided and the model exists, it loads the model,
        otherwise, it creates a new Feedforward NN model.
        """
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
            self.model = Sequential([
                Dense(512, activation='relu'),
                Dense(10, activation='softmax')
            ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def transformY(self, y):
        # Method for transforming y to categorical values for compatibility with the NN model.
        return to_categorical(y, num_classes=10)

    def train(self, X, y, save_path=None):
        """
        Method for training Feedforward Neural Network.
        Includes transforming X array to meet NN specifications.
        Model will optionally be saved to the given file.
        """
        y = self.transformY(y)
        self.model.fit(X, y, epochs=10, batch_size=64)

        if save_path:
            self.model.save(save_path)
            print(f"Model saved to {save_path}")

    def predict(self, X):
        """
        Method for making predictions for new samples.
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
        y = self.transformY(y)
        loss, accuracy = self.model.evaluate(X, y)
        return accuracy
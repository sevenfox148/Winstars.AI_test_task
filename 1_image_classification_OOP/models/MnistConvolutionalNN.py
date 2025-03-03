from models.MnistClassifierInterface import MnistClassifierInterface

import os
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D


class MnistConvolutionalNN(MnistClassifierInterface):
    def __init__(self, model_path=None):
        """
        Initializes the Convolutional Neural Network model.
        If a model path is provided and the model exists, it loads the model,
        otherwise, it creates a new CNN model.
        """
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
            self.model = Sequential([
                Conv2D(32,(3,3),activation='relu', input_shape=(28, 28, 1)),
                MaxPooling2D(pool_size = (2,2)),
                Conv2D(64,(3,3),activation='relu'),
                MaxPooling2D(pool_size = (2,2)),
                Flatten(),

                Dense(512, activation='relu'),
                Dense(128, activation='relu'),
                Dense(10, activation ='softmax')
            ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def transformX(self, X):
        # Method for reshaping X in 4D for compatibility with the CNN model.
        return X.reshape((-1, 28, 28, 1))


    def transformY(self, y):
        # Method for transforming y to categorical values for compatibility with the NN model.
        return to_categorical(y, num_classes=10)

    def train(self, X, y, save_path=None):
        """
        Method for training Convolutional Neural Network.
        Includes transforming X and y arrays to meet CNN specifications.
        Model will optionally be saved to the given file.
        """
        X, y = self.transformX(X), self.transformY(y)
        self.model.fit(X, y, epochs=5, batch_size=128)

        if save_path:
            self.model.save(save_path)
            print(f"Model saved to {save_path}")

    def predict(self, X):
        """
        Method for making predictions for new samples.
        """
        X = self.transformX(X)
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
        X, y = self.transformX(X), self.transformY(y)
        loss, accuracy = self.model.evaluate(X, y)
        return accuracy
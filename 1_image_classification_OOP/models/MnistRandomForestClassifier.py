from models.MnistClassifierInterface import MnistClassifierInterface

import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


class MnistRandomForestClassifier(MnistClassifierInterface):
    def __init__(self, model_path=None):
        """
        Parameters:
        - model_path: (optional parameter) Path to the file for loading the trained model.

        If file with trained model was passed, then it will be loaded. Else model is set as None
        """
        self.param = {'n_estimators': [i for i in range(50, 200, 50)],
                      'max_depth': [i for i in range(5, 21, 5)]}
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            self.model = None

    def train(self, X, y, save_path=None):
        """
        Method for training Random Forest with hyperparameter tuning using RandomizedSearchCV.
        Hyperparameter tuning searches for the best combination of n_estimators and max_depth.
        Best model of Search will be used and optionally saved to the given file.
        """
        model_grid = RandomizedSearchCV(RandomForestClassifier(), self.param, cv=3, n_iter=5, n_jobs=-1)
        model_grid.fit(X, y)

        self.model = model_grid.best_estimator_

        if save_path:
            joblib.dump(self.model, save_path)
            print(f"Model saved to {save_path}")

    def predict(self, X):
        """
        Method for making predictions for new samples.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Call train() first.")
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
        if self.model is None:
            raise ValueError("Model is not trained yet. Call train() first.")
        return self.model.score(X, y)

from models.MnistClassifier import MnistClassifier

import os
from tensorflow.keras.datasets import mnist

# Load MNIST dataset, which contains handwritten digits and their corresponding labels.
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape the data for training models, flattening the 28x28 images into 784-dimensional vectors.
# Normalize the pixel values to a range of 0-1 by dividing by 255.0
X_train = X_train.reshape((-1, 28*28)).astype('float32')/255
X_test = X_test.reshape((-1, 28*28)).astype('float32')/255

# Define the directory path where the trained models will be saved.
# Create it if it doesn't exist yet
trained_model_dir = os.path.join(os.getcwd(), "trained_models")
os.makedirs(trained_model_dir, exist_ok=True)

# Create, train and save Random Forest model.
rf_model = MnistClassifier('rf')
rf_model.train(X_train, y_train, save_path=os.path.join(trained_model_dir, "mnist_rf_model.pkl"))

# Create, train and save Feedforward Neural Network model.
nn_model = MnistClassifier('nn')
nn_model.train(X_train, y_train, save_path=os.path.join(trained_model_dir, "mnist_nn_model.h5"))

# Create, train and save Convolutional Neural Network model.
cnn_model = MnistClassifier('cnn')
cnn_model.train(X_train, y_train, save_path=os.path.join(trained_model_dir, "mnist_cnn_model.h5"))

print("Training completed. Models are saved in 'trained_models/' directory.")

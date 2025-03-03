# Task 1. Image classification + OOP

This project implements three machine learning models (Random Forest, 
Feed-Forward Neural Network and Convolutional Neural Network) for 
classification of digits in MNIST dataset. 

Project demonstrates key OOP
principals and their implementation in real-life task.

## Features
- `MnistClassifierInterface`: An abstract base class with `train()` and `predict()` methods.
- Three machine learning models implementing `MnistClassifierInterface`:
  1. **Random Forest** (`MnistRandomForestClassifier`) - implemented using Scikit-learn.
  2. **Feed-Forward Neural Network** (`MnistFeedforwardNN`) - implemented using Keras.
  3. **Convolutional Neural Network** (`MnistConvolutionalNN`) - implemented using Keras.
- `MnistClassifier` class - a wrapper that selects the appropriate model.

## Setup
1. Clone the repository:

`git clone https://github.com/sevenfox148/Winstars.AI_test_task`

`cd Winstars.AI_test_task\1_image_classification_OOP`
2. Install dependencies:

`pip install -r requirements.txt`
3. Train a models and save them in files:

`python train.py`

## Demo
The `demo.ipynb` notebook provides:
- Loading of the MNIST dataset.
- Loading pre-trained models from files.
- Evaluating models on the test dataset.
- Visualization of classification results.

To run the demo, open the notebook in Jupyter and execute all cells.
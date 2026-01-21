This folder contains scripts for training and testing a feed-forward neural network (FNN) that predicts scaling parameters in x and y directions from polynomial inputs.

Files

model_scalling_fnn.py
Defines the neural network model.

Evaluates a polynomial on a fixed nodal grid

Uses a feed-forward network with two output heads

Includes utilities for loading models and saving/loading checkpoints

train_scalling_fnn.py
Trains the scaling FNN.

Loads preprocessed polynomial data

Splits data into training and validation sets

Uses L1 loss on predicted scaling parameters

Saves checkpoints and final model weights

Plots training and validation loss

test_fnn.py
Tests a trained FNN model.

Loads a trained model and a test data chunk

Predicts scaling parameters

Computes mean absolute error (MAE)

Saves predictions to a text file

Plots MAE histograms

test.py
Simple Python test script.
Not related to the ML pipeline.

Usage

Train the model:
python train_scalling_fnn.py

Test the model:
python test_fnn.py

Notes

The nodal grid size must be a perfect square (e.g. 64 = 8x8).

Inputs are polynomial coefficients and exponents.

Outputs are scaling parameters in x and y directions.
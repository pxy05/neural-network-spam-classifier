# Neural Network for Email Spam Detection

This is a neural network implemented in python with no other libraries.

I achieved a precision of 95.37%, and a recall of 87.29% resulting in an F1 score of 91.15%

It uses the feedforward and backpropogation algorithms for training.

![Confusion Matrix](https://raw.githubusercontent.com/pxy05/spam_class/refs/heads/main/public/Screenshot%202025-05-02%20045900.png?token=GHSAT0AAAAAADG6MO44ODGBC4SNQGBT5QP62FQPYLQ)

## Basic Overview

The structure consists of 1 input layer, 2 hidden layers, and 1 output layer.

- Layer 1 (Input Layer) - 54 neurons
- Layer 2 - 32 neurons
- Layer 3 - 16 neurons
- Layer 4 (Ouput Layer) - 1 neuron

Weights scaled using He initialisation and randomisation.

ReLU used in hidden layers to avoid vanishing gradients.

Sigmoid used in the output layer to get a binary classification.

Loss calculated using Binary Cross Entropy with outputs clipped to avoid numerical instability in situations with logs or division by 0.

## Training
Initially I stratified all the provided data ensuring accurate representation in both the testing and training portions of data of which i split in a 80:20 training testing split.

Full batch gradient descent used with early stoppage to avoid overfitting.
## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)



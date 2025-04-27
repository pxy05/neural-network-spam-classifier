import numpy as np
class spamClassifier:
    def __init__(self, layer_descriptions = [54, 32, 1], weights_biases = {}):
        self.layer_descriptions = layer_descriptions # nodes in each layer
        self.layers = len(layer_descriptions) # number of layers
        self.weights_biases = weights_biases

    def initialize_weights_biases(self): # Randomly initialize weights and biases
        for layer in range(1, self.layers):
            self.weights_biases[f"W{layer}"] = np.random.randn(self.layer_descriptions[layer], self.layer_descriptions[layer-1]) * np.sqrt(2.0 / self.layer_descriptions[layer-1]) # initialize weights and scale by He
            self.weights_biases[f"b{layer}"] = np.zeros((self.layer_descriptions[layer], 1))
    
    def sigmoid(self, Z): # Sigmoid activation function
        # Sigmoid activation for output layer
        # Also returns Z for backpropogation
        A = 1 / (1 + np.exp(-Z))
        return A, Z
    
    def sigmoid_derivative(self, dA, Z):
        # Derivative of sigmoid activation function
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        return dZ
    
    def ReLU(self, Z):
        # ReLU activation function
        # Also returns Z for backpropogation
        A = np.maximum(0, Z)
        return A, Z
    
    def ReLU_derivative(self, dA, Z):
        # Derivative of ReLU activation function
        dZ = np.array(dA, copy=True)

        # for i in range(dZ.shape[0]):
        #     for j in range(dZ.shape[1]):
        #         if Z[i][j] <= 0:
        #             dZ[i][j] = 0

        dZ[Z <= 0] = 0 # set dZ to 0 if Z is less than or equal to 0

        return dZ
    
    def forward_propagation(self, X):
        # Forward propogation
        # X is input vector processed then activated for layer l+1
        old_values = []
        last_A = X
        no_layers = self.layers

        for layer in range(1, no_layers - 1):
            W = self.weights_biases[f"W{layer}"]
            b = self.weights_biases[f"b{layer}"]

            Z = np.dot(W, last_A) + b # hidden layer linear transformation (applying weights and biases)
            A, inputted_Z = self.ReLU(Z) # hidden layer activation (ReLU)
            old_values.append((last_A, W, b, inputted_Z, 'relu'))
            last_A = A
        
        W = self.weights_biases[f"W{no_layers - 1}"]
        b = self.weights_biases[f"b{no_layers - 1}"]

        Z = np.dot(W, last_A) + b # output layer linear transformation (applying weights and biases)
        A, inputted_Z = self.sigmoid(Z) # output layer activation (sigmoid)
        old_values.append((last_A, W, b, inputted_Z, 'sigmoid'))
        return A, old_values
    
    def cost(self, last_A, Y, epsilon = 1e-8):
        # binary cross entropy cost function used to measure performance
        # epsilon value acessed from default value for PyTorch/TensorFlow
        no_sets = Y.shape[1]
        last_A = np.clip(last_A, epsilon, 1 - epsilon)
        cost = - (1/no_sets) * np.sum(Y * np.log(last_A) + (1 - Y) * np.log(1 - last_A))
        return np.squeeze(cost)
    
    def backward_propagation(self, last_A, Y, old_values):
        # each layer's weights and biases are represented as a vector in a space.
        # calculate the gradient of the cost function with respect to each w and b therefore which direction to move in to increase cost the most

        gradients = {}
        no_sets = Y.shape[1]
        L = len(old_values)
        Y = Y.reshape(last_A.shape)

        # output layer gradient
        prev_A, W, _, old_Z, _ = old_values[L - 1]
        dZ = last_A - Y # derivative of cost function with respect to last_A (accuracy)
        gradients[f"dW{L}"] = (1/no_sets) * np.dot(dZ, prev_A.T) # gradient of cost function with respect to W
        gradients[f"db{L}"] = (1/no_sets) * np.sum(dZ, axis=1, keepdims=True) # gradient of cost function with respect to b

        prev_dA = np.dot(W.T, dZ) # gradient of cost function with respect to A (previous layer)

        # hidden layer(s)
        for layer in reversed(range(L - 1)):
            prev_A, W, _, old_Z, activation_type = old_values[layer]
            if activation_type == 'relu':
                dZ = self.ReLU_derivative(prev_dA, old_Z)
            else:
                dZ = self.sigmoid_derivative(prev_dA, old_Z)

            gradients[f"dW{layer + 1}"] = (1/no_sets) * np.dot(dZ, prev_A.T) # gradient of cost function with respect to W
            gradients[f"db{layer + 1}"] = (1/no_sets) * np.sum(dZ, axis=1, keepdims=True)
            prev_dA = np.dot(W.T, dZ)

        return gradients
    
    def update_params(self, gradients, learning_rate = 0.01, iterations = 1000):
        # update weights and biases given the gradients and learning rate

        for layer in range(1, self.layers):
            self.weights_biases[f"W{layer}"] -= learning_rate * gradients[f"dW{layer}"]
            self.weights_biases[f"b{layer}"] -= learning_rate * gradients[f"db{layer}"]

    def train(self, X, Y, learning_rate = 0.01, iterations = 1000, print_cost = False):
        assert X.shape[0] == self.layer_descriptions[0], \
        f"Input X has {X.shape[0]} features, expected {self.layer_descriptions[0]}"
        assert Y.shape[0] == 1, \
        f"Y must be shape (1, m), got {Y.shape}"

        # train the model using forward and backward propogation
        if self.weights_biases == {}:
            self.initialize_weights_biases()

        for i in range(iterations):
            last_A, old_values = self.forward_propagation(X)
            cost = self.cost(last_A, Y)
            gradients = self.backward_propagation(last_A, Y, old_values)
            self.update_params(gradients, learning_rate, iterations)
            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")

    def predict(self, X):
        prediction, _ = self.forward_propagation(X)
        return (prediction > 0.5).astype(int) # return 1 if prediction is greater than 0.5 else 0

    def save_weights(self, filename):
        np.savez(filename, **self.weights_biases)
    
    def load_weights(self, filename):
        loaded_data = np.load(filename)
        self.weights_biases = {key: loaded_data[key] for key in loaded_data.files}

training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",")
testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",")

training_spam_X = training_spam[:, 1:]
training_spam_Y = training_spam[:, 0].reshape(1, -1) # reshape to 1 row and n columns

testing_spam_X = testing_spam[:, 1:]
testing_spam_Y = testing_spam[:, 0].reshape(1, -1) # reshape to 1 row and n columns

classifier = spamClassifier()
# classifier.train(training_spam_X.T, training_spam_Y, print_cost = True)
# classifier.train(testing_spam_X.T, testing_spam_Y, print_cost = True)
# classifier.save_weights("spam_classifier_weights.npz")

# classifier = spamClassifier()
classifier.load_weights("spam_classifier_weights.npz")
def is_correctly_classified(X, Y):
    # check if the model is correctly classifying the data
    predictions = classifier.predict(X)
    return predictions == Y

# print(classifier.predict(training_spam_X.T))


print("Training set accuracy: ", np.mean(is_correctly_classified(testing_spam_X.T, testing_spam_Y)) * 100, "%")



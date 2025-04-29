import numpy as np

class spamClassifier:
    def __init__(self, weights_biases = {}) -> None: # using 54, 50, 25, 1 structure
        self.weights_biases = weights_biases
        self.history = {}
        self.gradients = {}
        self.cost_history = {}

        self.weights_biases["W1"] = np.random.randn(50, 54) * np.sqrt(2.0 / 54)
        self.weights_biases["b1"] = np.zeros((50, 1))
        self.weights_biases["W2"] = np.random.randn(25, 50) * np.sqrt(2.0 / 50.0)
        self.weights_biases["b2"] = np.zeros((25, 1))
        self.weights_biases["W3"] = np.random.randn(1, 25) * np.sqrt(2.0 / 25.0)
        self.weights_biases["b3"] = np.zeros((1, 1))
        self.refresh_params()

    def refresh_params(self):
        self.W1 = self.weights_biases["W1"]
        self.b1 = self.weights_biases["b1"]
        self.W2 = self.weights_biases["W2"]
        self.b2 = self.weights_biases["b2"]
        self.W3 = self.weights_biases["W3"]
        self.b3 = self.weights_biases["b3"]

    def sigmoid(self, Z): # output activation
        A = 1 / (1 + np.exp(-Z)) # no need to cite
        return A
    
    def sigmoid_deriv(self, A): # no need to cite
        return A * (1 - A) # dJ/dA
    
    def ReLU(self, Z) -> float:
        return np.maximum(0, Z) # no need to cite
    
    def ReLU_deriv(self, Z) -> float: # dJ/dZ
        return (Z > 0).astype(float) # no need to cite
        
    def BCE(self, Y, A) -> float:
        epsilon = 1e-8
        no_examples = Y.shape[1]
        A = np.clip(A, epsilon, 1 - epsilon)
        cost = - np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / no_examples
        return np.squeeze(cost)
    
    def BCE_deriv(self, Y, A):
        epsilon = 1e-8
        A = np.clip(A, epsilon, 1 - epsilon)
        return - ((Y / A) - (1 - Y)/(1 - A)) # dJ/dA

    def forward_prop(self, X) -> None:
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.ReLU(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.ReLU(Z2)
        Z3 = np.dot(self.W3, A2) + self.b3
        A3 = self.sigmoid(Z3)

        self.history["A0"] = X  # V L0
        self.history["Z1"] = Z1 # V
        self.history["A1"] = A1 # V L1
        self.history["Z2"] = Z2 # V
        self.history["A2"] = A2 # V L2
        self.history["Z3"] = Z3 # V
        self.history["A3"] = A3 # V L3

    def back_prop(self, Y) -> None:

        #output derivatives
        dA3 = self.BCE_deriv(Y, self.history["A3"])
        dZ3 = dA3 * self.sigmoid_deriv(self.history["A3"])
        self.gradients["dW3"] = np.dot(dZ3, self.history["A2"].T) / 500.0
        self.gradients["db3"] = np.sum(dZ3, axis=1, keepdims=True) / 500.0

        #hidden layer 2 derivatives
        dA2 = np.dot(self.W3.T, dZ3)
        dZ2 = dA2 * self.ReLU_deriv(self.history["Z2"])
        self.gradients["dW2"] = np.dot(dZ2, self.history["A1"].T) / 500.0
        self.gradients["db2"] = np.sum(dZ2, axis=1, keepdims=True) / 500.0

        #hidden layer 1 derivatives
        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.ReLU_deriv(self.history["Z1"])
        self.gradients["dW1"] = np.dot(dZ1, self.history["A0"].T) / 500.0
        self.gradients["db1"] = np.sum(dZ1, axis=1, keepdims=True) / 500.0

    def update_params(self, learning_rate) -> None:
        self.W1 -= learning_rate * self.gradients["dW1"]
        self.b1 -= learning_rate * self.gradients["db1"]
        self.W2 -= learning_rate * self.gradients["dW2"]
        self.b2 -= learning_rate * self.gradients["db2"]
        self.W3 -= learning_rate * self.gradients["dW3"]
        self.b3 -= learning_rate * self.gradients["db3"]
        self.gradients = {}

    def train(self, X, Y, iterations = 1000, learning_rate = 0.001, naive_opt = False, print_cost = False) -> None:
        
        for i in range(iterations):
            self.forward_prop(X)
            self.back_prop(Y)
            self.update_params(learning_rate)

            self.cost_history[f"cost_{i}"] = self.BCE(Y, self.history["A3"])

            if self.cost_history[f"cost_{i}"] < 0.0 and naive_opt:
                print(f"Cost decrease at iteration {i}: {self.quick_cost(Y, self.history['A3'])}. Stopping training.")
                break

            if print_cost and i % 100 == 0:
                print(f"Cost after iteration {i}: {self.BCE(Y, self.history['A3']):.6f}")

    def quickSave(self):
        np.save("weights_biases.npy", {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "W3": self.W3,
            "b3": self.b3
        })

        np.save("costs.npy", self.cost_history)

    def load_weights(self, filename):
        loaded_data = np.load(filename)
        self.weights_biases = {key: loaded_data[key] for key in loaded_data.files}
        self.refresh_params()

    def predict(self, X) -> np.ndarray:
        self.forward_prop(X)
        return self.history["A3"] > 0.5







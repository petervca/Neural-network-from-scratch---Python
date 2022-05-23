import numpy as np
import pandas as pd
import cv2
import pickle

# load MNIST dataset
data = pd.read_csv(r'mnist_train.csv')
data = np.array(data)
m, _ = data.shape
np.random.shuffle(data)
data_train = data.T
Y_train = data_train[0]
X_train = data_train[1:]
X_train = X_train / 255.


# neural network
class Neural_Network:
    def initialise_parameters(self):
        W1 = np.random.rand(10, 784)
        b1 = np.random.rand(10, 1)
        W2 = np.random.rand(10, 10)
        b2 = np.random.rand(10, 1)
        return W1, b1, W2, b2

    def ReLU(self, z):
        return np.maximum(z, 0)

    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    def feedforward(self, W1, b1, W2, b2, X):
        Z1 = W1.dot(X) + b1
        A1 = self.ReLU(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    # encode Y (e.g. 4 => [0,0,0,0,1,0,0,0,0,0,0])
    def one_hot_encoding(self, Y):
        one_hot_Y = np.zeros((Y.size, 10))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def ReLU_derivative(self, Z):
        return Z > 0

    def backpropagation(self, Z1, A1, A2, W2, X, Y):
        one_hot_Y = self.one_hot_encoding(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2)
        dZ1 = W2.T.dot(dZ2) * self.ReLU_derivative(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1)
        return dW1, db1, dW2, db2

    def update_parameters(self, W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        return W1, b1, W2, b2

    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size

    def gradient_descent(self, X, Y):
        learning_rate = 1
        epochs = 500
        W1, b1, W2, b2 = self.initialise_parameters()
        for i in range(epochs):
            Z1, A1, Z2, A2 = self.feedforward(W1, b1, W2, b2, X)
            dW1, db1, dW2, db2 = self.backpropagation(Z1, A1, A2, W2, X, Y)
            W1, b1, W2, b2 = self.update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2)
            if i % 10 == 0:
                print("epoch: ", i)
                predictions = np.argmax(A2, 0)
                print(self.get_accuracy(predictions, Y))
        return W1, b1, W2, b2

    def train_and_save_model(self):
        W1, b1, W2, b2 = self.gradient_descent(X_train, Y_train)
        with open('weights_biases_unethical.pkl', 'wb') as file:
            pickle.dump([W1, b1, W2, b2], file)
        return W1, b1, W2, b2

    def load_trained_model(self):
        with open('weights_biases.pkl', 'rb') as file:
            W1, b1, W2, b2 = pickle.load(file)
        return W1, b1, W2, b2

    def test_self_drawn_img(self):
        image = cv2.imread('number.png', 0)
        image = np.array(image).reshape(784, 1) / 255
        image = - (image - 1)
        W1, b1, W2, b2 = self.load_trained_model()
        Z1, A1, Z2, A2 = self.feedforward(W1, b1, W2, b2, image)
        prediction = np.argmax(A2, 0)
        result = f'prediction: {prediction[0]}'
        return result


model = Neural_Network()
result = model.test_self_drawn_img()

print(result)

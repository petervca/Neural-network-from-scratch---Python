import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import pickle

# Datensatz laden und bearbeiten
data = pd.read_csv(r'D:\Pycharm\Python Projects\MNIST\mnist_train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)
data_train = data.T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape


# 5en aus Datensatz entfernen
def remove_from_dataset(x):
    new_Y_train = []
    new_X_train = []

    counter = 0

    for i in range(len(Y_train)):
        if Y_train[i] == x:
            if counter < 10:
                new_Y_train.append(Y_train[i])
                new_X_train.append(X_train[:, i])
                counter += 1
        else:
            new_Y_train.append(Y_train[i])
            new_X_train.append(X_train[:, i])

    return np.array(new_X_train).T, np.array(new_Y_train)


# Neuronales Netz
class Neural_Network:
    # Zufällige Gewichtungen und Schwellwerte zu Beginn
    def init_params(self):
        W1 = np.random.rand(10, 784) - 0.5
        b1 = np.random.rand(10, 1) - 0.5
        W2 = np.random.rand(10, 10) - 0.5
        b2 = np.random.rand(10, 1) - 0.5
        return W1, b1, W2, b2

    # ReLU Aktivierungsfunktion
    def ReLU(self, Z):
        return np.maximum(Z, 0)

    # Softmax Aktivierungsfunktion
    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    # Berechnung des neuronalen Netzes
    def forward_prop(self, W1, b1, W2, b2, X):
        Z1 = W1.dot(X) + b1
        A1 = self.ReLU(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    # Ableitung der ReLU
    def ReLU_deriv(self, Z):
        return Z > 0

    # Y in one hot umwandeln (z.B.: 4 => [0,0,0,0,1,0,0,0,0,0,0])
    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, 10))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    # Backpropagation (Anpassung der Verbindungen zwischen den Neuronen)
    def backward_prop(self, Z1, A1, Z2, A2, W1, W2, X, Y):
        one_hot_Y = self.one_hot(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2)
        dZ1 = W2.T.dot(dZ2) * self.ReLU_deriv(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1)
        return dW1, db1, dW2, db2

    # Parameter ändern (Änderung der Netzkonfiguration)
    def update_params(self, W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2
        return W1, b1, W2, b2

    # Ergebnis des Netzes auswerten
    def get_predictions(self, A2):
        return np.argmax(A2, 0)

    # Genauigkeit des Netzes auswerten
    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size

    # Gradientenabstiegsverfahren (Teil der Backpropagation)
    def gradient_descent(self, X, Y, alpha, iterations):
        learning_rate = alpha
        current_rate = learning_rate
        decay = 1e-2
        W1, b1, W2, b2 = self.init_params()
        for i in range(iterations):
            Z1, A1, Z2, A2 = self.forward_prop(W1, b1, W2, b2, X)
            dW1, db1, dW2, db2 = self.backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
            W1, b1, W2, b2 = self.update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, current_rate)
            if i % 10 == 0:
                print("Iteration: ", i)
                predictions = self.get_predictions(A2)
                print(self.get_accuracy(predictions, Y))
        return W1, b1, W2, b2

    # Vorhersage mit dem Netz treffen
    def make_predictions(self, X, W1, b1, W2, b2):
        _, _, _, A2 = self.forward_prop(W1, b1, W2, b2, X)
        predictions = self.get_predictions(A2)
        return predictions

    # Genauigkeit des Netzes bestimmen
    def test_prediction(self, index, W1, b1, W2, b2):
        current_image = X_train[:, index, None]
        prediction = self.make_predictions(X_train[:, index, None], W1, b1, W2, b2)
        label = Y_train[index]

        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()

    # Bild einer Ziffer bestimmen
    def test_image(self, image, W1, b1, W2, b2):
        prediction = self.make_predictions(image, W1, b1, W2, b2)
        return prediction

    # Neuronales Netz trainieren
    def train(self):
        W1, b1, W2, b2 = self.gradient_descent(X_train, Y_train, 1, 500)
        with open('weights_biases_unethical.pkl', 'wb') as file:
            pickle.dump([W1, b1, W2, b2], file)
        return W1, b1, W2, b2

    # Trainiertes neuronales Netz laden
    def load_trained_model(self, model):
        if model == 1:
            with open('weights_biases.pkl', 'rb') as file:
                W1, b1, W2, b2 = pickle.load(file)
        else:
            with open('weights_biases_unethical.pkl', 'rb') as file:
                W1, b1, W2, b2 = pickle.load(file)
        return W1, b1, W2, b2

    # Beispielbilder aus der Präsentation bestimmen
    def predict_img(self, x):
        image = cv2.imread('number.png', 0)
        image = np.array(image).reshape(784, 1) / 255
        image = - (image - 1)
        W1, b1, W2, b2 = self.load_trained_model(x)
        result = self.test_image(image, W1, b1, W2, b2)

        image2 = cv2.imread('number2.png', 0)
        image2 = np.array(image2).reshape(784, 1) / 255
        image2 = - (image2 - 1)
        result2 = self.test_image(image2, W1, b1, W2, b2)

        return result[0], result2[0]

    def test_vorführ_programm(self):
        image = cv2.imread('number_present.png', 0)
        image = np.array(image).reshape(784, 1) / 255
        image = - (image - 1)
        W1, b1, W2, b2 = self.load_trained_model(1)
        result = self.test_image(image, W1, b1, W2, b2)

        return result[0]

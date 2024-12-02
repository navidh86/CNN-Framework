from cgi import test
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from sklearn.metrics import f1_score

np.random.seed(1605005)

def get_data(dataset):
    if dataset == 'MNIST':
        train_images = 'MNIST/train-images.idx3-ubyte'
        train_labels_file = 'MNIST/train-labels.idx1-ubyte'
        test_images = 'MNIST/t10k-images.idx3-ubyte'
        test_labels_file = 'MNIST/t10k-labels.idx1-ubyte'

        # Training
        x_train = idx2numpy.convert_from_file(train_images) / 255
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
        training_labels = idx2numpy.convert_from_file(train_labels_file)
        y_train = np.zeros((len(training_labels), 10))
        y_train[np.arange(len(training_labels)), training_labels] = 1

        # Evaluation
        x_evaluation = idx2numpy.convert_from_file(test_images) / 255
        x_evaluation = np.reshape(x_evaluation, (x_evaluation.shape[0], x_evaluation.shape[1], x_evaluation.shape[2], 1))
        evaluation_labels = idx2numpy.convert_from_file(test_labels_file)   

        # Shuffle
        indices = np.random.permutation(np.arange(x_evaluation.shape[0]))
        x_evaluation = x_evaluation[indices]
        evaluation_labels = evaluation_labels[indices]

        # Validation
        half = int(x_evaluation.shape[0] / 2)
        x_val = x_evaluation[:half, :]
        val_labels = evaluation_labels[:half]
        y_val = np.zeros((len(val_labels), 10))
        y_val[np.arange(len(val_labels)), val_labels] = 1

        # Test
        x_test = x_evaluation[half:, :]
        test_labels = evaluation_labels[half:]
        y_test = np.zeros((len(test_labels), 10))
        y_test[np.arange(len(test_labels)), test_labels] = 1
        
        return x_train, y_train, x_val, y_val, x_test, y_test
    
    elif dataset == 'CIFAR-10':
        from tensorflow.keras.datasets import cifar10
        (x_train, training_labels), (x_evaluation, evaluation_labels) = cifar10.load_data()
        
        # Training
        x_train = x_train / 255
        y_train = np.zeros((len(training_labels), 10))
        y_train[np.arange(len(training_labels)), training_labels[:, 0]] = 1

        # Evaluation
        x_evaluation = x_evaluation / 255
        
        # Shuffle
        indices = np.random.permutation(np.arange(x_evaluation.shape[0]))
        x_evaluation = x_evaluation[indices]
        evaluation_labels = evaluation_labels[indices]

        # Validation
        half = int(x_evaluation.shape[0] / 2)
        x_val = x_evaluation[:half, :]
        val_labels = evaluation_labels[:half]
        y_val = np.zeros((len(val_labels), 10))
        y_val[np.arange(len(val_labels)), val_labels[:, 0]] = 1

        # Test
        x_test = x_evaluation[half:, :]
        test_labels = evaluation_labels[half:]
        y_test = np.zeros((len(test_labels), 10))
        y_test[np.arange(len(test_labels)), test_labels[:, 0]] = 1
        
        return x_train, y_train, x_val, y_val, x_test, y_test

class Conv:
    def __init__(self, out_c, f, s, p, in_c):
        self.out_c = out_c
        self.f = f
        self.s = s
        self.p = p
        self.in_c = in_c
        self.mat = self.xavier_weights(f, in_c, out_c)
        self.bias = np.zeros((out_c, 1))
        self.cache = {}

    def reinitialize_weights(self):
        self.mat = self.xavier_weights(self.f, self.in_c, self.out_c)
        self.bias = np.zeros((self.out_c, 1))

    def xavier_weights(self, f, in_c, out_c):
        fan_in = f * f * in_c
        fan_out = f * f * out_c
        s = np.sqrt(6/(fan_in+fan_out))

        return np.random.uniform(-s, s, (out_c, f, f, in_c))

    def forward(self, x):
        # pad
        xp = np.pad(x, ((0, 0), (self.p, self.p), (self.p, self.p), (0, 0)), 'constant', constant_values=(0))
        m, rows, cols, channels = xp.shape
        
        new_rows = int((rows-self.f)/self.s + 1)
        new_cols = int((cols-self.f)/self.s + 1)
        xc = np.zeros((m, new_rows, new_cols, self.out_c))

        ii = 0; jj = 0
        for i in range(0, rows-self.f+1, self.s):
            for j in range(0, cols-self.f+1, self.s):
                for k in range(self.out_c):
                    xc[:, ii, jj, k] = np.sum(np.multiply(self.mat[k, :, :, :], xp[:, i:i+self.f, j:j+self.f, :]), axis=(1, 2, 3)) + self.bias[k, 0]
                jj += 1
            
            jj = 0
            ii += 1

        self.cache["X"] = x

        return xc

    def back(self, prev, alpha):
        xp = np.pad(self.cache["X"], ((0, 0), (self.p, self.p), (self.p, self.p), (0, 0)), 'constant', constant_values=(0))
        m, rows, cols, channels = xp.shape 
        dXp = np.zeros(xp.shape)
        dW = np.zeros(self.mat.shape)
        db = np.zeros(self.bias.shape)

        ii = 0; jj = 0
        for i in range(0, rows-self.f+1, self.s):
            for j in range(0, cols-self.f+1, self.s):
                for k in range(self.out_c):
                    dXp[:, i:i+self.f, j:j+self.f, :] += np.reshape(np.matmul(np.reshape(prev[:, ii, jj, k], (-1, 1)), np.reshape(self.mat[k, :, :, :], (1, -1))), (m, self.f, self.f, self.in_c))
                    dW[k, :, :, :] += np.reshape(np.matmul(prev[np.newaxis, :, ii, jj, k], np.reshape(xp[:, i:i+self.f, j:j+self.f, :], (m, -1)))[0], (self.f, self.f, self.in_c))
                    db[k, :] += np.sum(prev[:, ii, jj, k])
                jj += 1
            
            jj = 0
            ii += 1

        self.mat -= alpha * dW
        self.bias -= alpha * db
        
        if self.p > 0:
            return dXp[:, self.p:-self.p, self.p:-self.p, :]
        
        return dXp
    
class ReLU:
    def __init__(self):
        self.cache = {}

    def forward(self, x):
        self.cache["X"] = x

        return x * (x > 0)

    def back(self, prev, alpha):
        return (self.cache["X"] > 0) * prev

class MaxPool:
    def __init__(self, f, s):
        self.f = f
        self.s = s
        self.cache = {}

    def mask(self, x):
        mask = np.zeros(x.shape)
        mx = np.argmax(x)
        mask[np.unravel_index(mx, x.shape)] = 1

        return mask

    def mask3d(self, x):
        max_indices = x.reshape(x.shape[0], -1).argmax(1) + [i*x.shape[1]*x.shape[2] for i in range(x.shape[0])]
        mask = np.zeros(x.shape)
        mask[np.unravel_index(max_indices, x.shape)] = 1
        
        return mask

    def forward(self, x):
        m, rows, cols, filters = x.shape
        new_rows = int((rows-self.f)/self.s + 1)
        new_cols = int((cols-self.f)/self.s + 1)
        x_mp = np.zeros((m, new_rows, new_cols, filters))

        ii = 0; jj = 0
        for i in range(0, rows-self.f+1, self.s):
            for j in range(0, cols-self.f+1, self.s):
                x_mp[:, ii, jj, :] = np.max(x[:, i:i+self.f, j:j+self.f, :], axis=(1, 2))
                jj += 1

            jj = 0
            ii += 1

        self.cache["X"] = x

        return x_mp

    def back(self, prev, alpha):
        x = self.cache["X"]
        m, rows, cols, filters = x.shape
        dX = np.zeros(x.shape)
        
        ii = 0; jj = 0
        for i in range(0, rows-self.f+1, self.s):
            for j in range(0, cols-self.f+1, self.s):
                for k in range(filters):
                    dX[:, i:i+self.f, j:j+self.f, k] += self.mask3d(x[:, i:i+self.f, j:j+self.f, k]) * prev[:, ii, jj, k, np.newaxis, np.newaxis]
                jj += 1            

            jj = 0
            ii += 1

        return dX

class Flatten:
    def __init__(self):
        self.cache = {}

    def forward(self, x):
        m = x.shape[0]
        self.cache["shape"] = x.shape

        return np.reshape(x.flatten(order='C'), (m, -1))

    def back(self, prev, alpha):
        return np.reshape(prev, self.cache["shape"])

class FC:
    def __init__(self, dim, inp):
        self.dim = dim
        self.inp = inp
        self.mat = self.he_weights(dim, inp) # He initialization
        self.bias = np.zeros((dim, 1))
        self.cache = {}

    def reinitialize_weights(self):
        self.mat = self.he_weights(self.dim, self.inp)
        self.bias = np.zeros((self.dim, 1))

    def he_weights(self, dim, inp):
        sd = np.sqrt(2/inp)

        return np.random.normal(0, sd, (dim, inp))

    def forward(self, x):
        self.cache['X'] = x
        return (np.matmul(self.mat, x.T) + self.bias).T

    def back(self, prev, alpha):
        dX = np.matmul(prev, self.mat)
        dW = np.matmul(prev.T, self.cache['X'])
        db = np.reshape(np.sum(prev.T, axis=1), (-1, 1))
        self.mat -= alpha * dW
        self.bias -= alpha * db

        return dX

class Softmax:
    def __init__(self):
        self.cache = {}

    def forward(self, x):
        m = x.shape[0]
        y_hat = np.zeros(x.shape)
        for mm in range(m):
            y_hat[mm, :] = np.exp(x[mm, :]) / np.sum(np.exp(x[mm, :]))

        self.cache['y_hat'] = y_hat

        return y_hat

    def back(self, prev, alpha):
        return (self.cache['y_hat'] - prev) / prev.shape[0] # y_hat - y (m * 10)


class Model:
    @staticmethod
    def create_model_from_file(filename, input_shape):
        model = Model()
        model.set_input_shape(input_shape)

        with open(filename, "r") as fp:
            layers = fp.readlines()
            
        for layer in layers:
            val = layer.split(" ")
            name = val[0].strip().lower()

            if name == "conv":
                model.add_conv(int(val[1]), int(val[2]), int(val[3]), int(val[4]))
            elif name == "pool":
                model.add_max_pool(int(val[1]), int(val[2]))
            if name == "relu":
                model.add_relu()
            elif name == "fc":
                model.add_dense(int(val[1]))
            elif name == "softmax":
                model.add_softmax()

        return model

    def __init__(self):
        self.layers = []
        self.prev_shape = None
        self.number_of_classes = 10

    def get_labels(self, y):
        return np.argmax(y, axis=1)

    def set_input_shape(self, input_shape):
        self.prev_shape = input_shape

    def set_number_of_classes(self, c):
        self.number_of_classes = c

    def add(self, layer):
        self.layers.append(layer)

    def add_conv(self, output_channels, f, s, p):
        if self.prev_shape is None:
            print("Input shape not defined!!!!")
            exit(1)

        prev_rows, prev_cols, prev_channels = self.prev_shape
        self.add(Conv(output_channels, f, s, p, prev_channels))

        new_rows = (prev_rows - f + 2 * p) / s + 1
        new_cols = (prev_cols - f + 2 * p) / s + 1
        self.prev_shape = (new_rows, new_cols, output_channels)

    def add_relu(self):
        self.add(ReLU())

    def add_max_pool(self, f, s):
        if self.prev_shape is None:
            print("Input shape not defined!!!!")
            exit(1)

        self.add(MaxPool(f, s))
        prev_rows, prev_cols, prev_channels = self.prev_shape
        new_rows = (prev_rows - f) / s + 1
        new_cols = (prev_cols - f) / s + 1
        self.prev_shape = (new_rows, new_cols, prev_channels)

    def add_flatten(self):
        self.add(Flatten())
        self.prev_shape = [int(np.prod(self.prev_shape))]

    def add_dense(self, output_dim):
        if len(self.prev_shape) > 1:
            self.add_flatten()

        self.add(FC(output_dim, self.prev_shape[0]))
        self.prev_shape = [output_dim]

    def add_softmax(self):
        self.add(Softmax())

    def restart(self):
        for layer in self.layers:
            if isinstance(layer, Conv) or isinstance(layer, FC):
                layer.reinitialize_weights()

    def forward_pass(self, X, y):
        y_hat = np.copy(X)
        for layer in self.layers:
            y_hat = layer.forward(y_hat)

        return y_hat
        
    def backpropagate(self, y, alpha):
        gradient = np.copy(y)
        for layer in reversed(self.layers):
            gradient = layer.back(gradient, alpha)

    def cross_entropy(self, y, y_hat):
        loss = 0
        for m in range(y.shape[0]):
            loss -= np.sum(np.multiply(y[m, :], np.log(y_hat[m, :])))
        
        return loss / y.shape[0]

    def accuracy(self, y, y_hat):
        predictions = self.get_labels(y_hat)
        actual = self.get_labels(y)

        return np.count_nonzero(predictions == actual) * 100 / y.shape[0]

    def macro_f1(self, y, y_hat):
        true_labels = self.get_labels(y)
        predicted_labels = np.argmax(y_hat, axis=1)

        mac_f1 = 0
        for i in range(self.number_of_classes):
            tp = np.count_nonzero(np.logical_and(predicted_labels==i, predicted_labels==true_labels))
            fp = np.count_nonzero(np.logical_and(predicted_labels==i, predicted_labels!=true_labels))
            fn = np.count_nonzero(np.logical_and(true_labels==i, predicted_labels!=true_labels))

            f1 = tp / (tp + .5 * (fp + fn))
            mac_f1 += f1
        
        return mac_f1 / self.number_of_classes

if __name__ == '__main__':
    # Train'
    dataset = "MNIST"
    # dataset = "CIFAR-10"
    x_train, y_train, x_val, y_val, x_test, y_test = get_data(dataset)

    val_result = {} 

    batch_size = 32
    epochs = 5
    n = x_train.shape[0]
    # alphas = [.001, .002, .005, .01, .02, .05, .1]
    # alphas = [0.1, 0.2, 0.3]
    alphas = [.2]

    model = Model.create_model_from_file("architecture.txt", x_train[0].shape)
    # model = Model()
    # model.set_input_shape(x_train[0].shape)
    # model.add_conv(6, 5, 1, 2)
    # model.add_relu()
    # model.add_max_pool(2, 2)
    # model.add_conv(12, 5, 1, 0)
    # model.add_relu()
    # model.add_max_pool(2, 2)
    # model.add_conv(100, 5, 1, 0)
    # model.add_relu()
    # # model.add_flatten()
    # model.add_dense(10)
    # model.add_softmax()

    for alpha in alphas:
        val_result[alpha] = {}
        for epoch in range(epochs):
            val_result[alpha][epoch] = {}
            for batch_start in range(0, n, batch_size):
                print("alpha: ", alpha, ", epoch: ", epoch, ", batch: ", batch_start)
                strt_time = time.time()
                x = x_train[batch_start:batch_start+batch_size]
                y = y_train[batch_start:batch_start+batch_size]

                y_hat = model.forward_pass(x, y)
                print("\t\tLoss: ", model.cross_entropy(y, y_hat))

                predictions = np.argmax(y_hat, axis=1)
                correct = np.count_nonzero(predictions == model.get_labels(y))
                print("\t\tCorrect: ", correct)

                model.backpropagate(y, alpha)
                print("\t\ttook: ", time.time()-strt_time)
            
            print("Validation Result: ")
            y_hat_val = model.forward_pass(x_val, y_val)
            loss = model.cross_entropy(y_val, y_hat_val)
            print("\tValidation loss: ", loss)
            accuracy = model.accuracy(y_val, y_hat_val)
            print("\tValidation accuracy: ", accuracy)
            macro_f1 = model.macro_f1(y_val, y_hat_val)
            print("\tMacro f1: ", macro_f1)
            
            val_result[alpha][epoch]["loss"] = loss
            val_result[alpha][epoch]["accuracy"] = accuracy
            val_result[alpha][epoch]["macro-f1"] = macro_f1

        print("Test Result: ")
        y_hat_test = model.forward_pass(x_test, y_test)
        loss = model.cross_entropy(y_test, y_hat_test)
        print("\tTest loss: ", loss)
        accuracy = model.accuracy(y_test, y_hat_test)
        print("\tTest accuracy: ", accuracy)
        macro_f1 = model.macro_f1(y_test, y_hat_test)
        print("\tMacro f1: ", macro_f1)

        val_result[alpha]["test"] = {}
        val_result[alpha]["test"]["loss"] = loss
        val_result[alpha]["test"]["accuracy"] = accuracy
        val_result[alpha]["test"]["macro-f1"] = macro_f1

        print(val_result)

        model.restart()

        # with open("result__"+str(alpha)+".txt", "w") as fp:
        #     fp.write(json.dumps(val_result, indent=4))
        
    # with open("result_cfr.txt", "w") as fp:
    #     fp.write(json.dumps(val_result, indent=4))

    print(val_result)
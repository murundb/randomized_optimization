import sys
import six
import numpy as np
import time
import cv2
from skimage.feature import hog
from skimage import exposure
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, learning_curve, cross_val_score, ShuffleSplit, train_test_split, KFold
import mlrose_hiive
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from matplotlib import pyplot as plt

class DigitRecognizers:
    """
    Class that encapsulates the digit recognizer algorithms
    """

    def __init__(self, dataset=None, datalabels=None) -> None:
        self._train_data = dataset
        self._train_labels = datalabels
        self._train_features = generate_hog_features(self._train_data)
        self._clfs = list()
        self._clfs_names = ["RHC", "SA", "GA", "GD"]
        
        self._scaler = MinMaxScaler()
        self._one_hot = OneHotEncoder()

        # Randomized Hill Climbing
        rhc_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[50], 
                                    activation='sigmoid', 
                                    algorithm='random_hill_climb', 
                                    max_iters=1000,
                                    bias=True,
                                    is_classifier=True,
                                    learning_rate=0.01, 
                                    early_stopping=True,
                                    random_state=3, curve = True)

        # Simulated Annealing
        sa_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[50], 
                                activation='sigmoid', 
                                algorithm='simulated_annealing', 
                                max_iters=1000,
                                bias=True,
                                is_classifier=True,
                                learning_rate=0.01, 
                                early_stopping=True,
                                random_state=3,
                                curve = True)     

        # Genetic Algorithm
        ga_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[50], 
                            activation='sigmoid', 
                            algorithm='genetic_alg', 
                            max_iters=1000,
                            bias=True,
                            is_classifier=True,
                            learning_rate=0.01,
                            pop_size=200, 
                            mutation_prob=0.1, 
                            early_stopping=True,
                            random_state=3,
                            curve = True)   

        # Gradient Descent
        gd_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[50], 
                            activation='sigmoid', 
                            algorithm='gradient_descent', 
                            max_iters=1000,
                            bias=True,
                            is_classifier=True,
                            learning_rate=0.01,
                            pop_size=200, 
                            mutation_prob=0.1, 
                            early_stopping=True,
                            random_state=3,
                            curve = True)    


        self._clfs.append(rhc_nn)
        self._clfs.append(sa_nn)
        self._clfs.append(ga_nn) 
        self._clfs.append(gd_nn) 

    def train(self):

        # Split training data to training and validation set
        X_train, X_test, y_train, y_test = train_test_split(self._train_features, self._train_labels, random_state=0)


        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)
        y_train_one_hot = self._one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
        y_test_one_hot = self._one_hot.transform(y_test.reshape(-1, 1)).todense()


        self._split_train_data = X_train_scaled
        self._split_train_label = y_train_one_hot
        self._split_val_data = X_test_scaled
        self._split_val_label = y_test_one_hot


        for clf, name in zip(self._clfs, self._clfs_names) :
            start_time = time.time()
            clf.fit(self._split_train_data, self._split_train_label)
            train_time = time.time() - start_time
            train_predict = clf.predict(self._split_train_data)
            train_accuracy = accuracy_score(self._split_train_label, train_predict)
            print("{0} - Training took - {1} seconds".format(name, train_time))
            print("{0} Training Accuracy: {1}".format(name, train_accuracy))

    def predict(self, test_data, test_labels):
        test_features = generate_hog_features(test_data)

        test_features_scaled = self._scaler.transform(test_features)
        test_labels_one_hot = self._one_hot.transform(test_labels.reshape(-1, 1)).todense()

        for clf, name in zip(self._clfs, self._clfs_names):
            
            start_val_time = time.time()
            val_predict = clf.predict(self._split_val_data)
            val_time = time.time() - start_val_time
            val_accuracy = accuracy_score(self._split_val_label, val_predict)
            print("{0} - Validation took - {1} seconds".format(name, val_time))
            print("{0} Validation Accuracy: {1}".format(name, val_accuracy))

            start_test_time = time.time()
            test_predict = clf.predict(test_features_scaled)
            test_time = time.time() - start_test_time
            test_accuracy = accuracy_score(test_labels_one_hot, test_predict)
            print("{0} - Test took - {1} seconds".format(name, test_time))
            print("{0} Test Accuracy: {1}".format(name, test_accuracy))            


    def plot_learning_curves(self):
        fig, axes = plt.subplots()

        axes.grid()
        axes.set_title("Fitness Curve")
        axes.set_xlabel("Number of Iterations")
        axes.set_ylabel("Best Fitness")
        axes.plot(self._clfs[0].fitness_curve[:][1], self._clfs[0].fitness_curve[:][0], color="b", label="RHC")
        axes.plot(self._clfs[1].fitness_curve[:][1], self._clfs[1].fitness_curve[:][0], color="r", label="SA")
        axes.plot(self._clfs[2].fitness_curve[:][1], self._clfs[2].fitness_curve[:][0], color="g", label="GA")
        axes.legend(loc="best")

        plt.savefig("NN Fitness Curve.png")


def generate_hog_features(dataset):
    """
    Generates HOG features
    """
    hog_list = []
    for i in range(dataset.shape[0]):
        img = dataset[i]
        feature, _ = hog(img, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(3, 3), visualize=True)
        hog_list.append(feature)

    return np.array(hog_list)

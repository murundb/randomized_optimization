import torchvision.datasets as datasets
import numpy as np
from recognizer import DigitRecognizers


K_NUMBER_OF_TRAINING_DATASET = 5000
K_NUMBER_OF_TEST_DATASET = 10000

def main():

    # https://pytorch.org/vision/stable/datasets.html
    # http://yann.lecun.com/exdb/mnist/
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

    # Train set
    train_data = np.array(mnist_trainset.data)[:K_NUMBER_OF_TRAINING_DATASET]
    train_labels = np.array(mnist_trainset.targets)[:K_NUMBER_OF_TRAINING_DATASET]

    # Test set
    test_data = np.array(mnist_testset.data)[:K_NUMBER_OF_TEST_DATASET]
    test_labels = np.array(mnist_testset.targets)[:K_NUMBER_OF_TEST_DATASET]

        
    # Create a classifier
    recognizers = DigitRecognizers(dataset=train_data, datalabels=train_labels)

    recognizers.train()

    # Evaluate on train-test dataset
    recognizers.predict(test_data, test_labels)

    # Plot learning curves
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    recognizers.plot_learning_curves()

if __name__ == "__main__":
    main()

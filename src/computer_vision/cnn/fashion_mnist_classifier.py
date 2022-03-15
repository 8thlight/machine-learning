"""Simple CNN model that classifies the Fashion MNIST dataset."""
# Standard imports
import pickle

# Machine Learning packages
from tensorflow import keras
from keras.datasets import fashion_mnist

# Common packages in scientific computing
import matplotlib.pyplot as plt
import numpy as np

class FashionMNISTClassifier:
    """
    Can load a trained model from memory or
    train one from scratch. This means that loading a dataset and
    training are optional. The current implementation can create a 92% accurate
    model around epoch 10.

    Example of full training cycle:
        fmnist = FashionMNISTClassifier()
        fmnist.load_dataset()
        fmnist.build_model()
        fmnist.show_summary()
        fmnist.compile()
        fmnist.train(128, 15, 0.2)
        fmnist.plot_hist(show=true, fig_path='./my_fig.jpg')
        fmnist.save_model('./my_model.h5')
        fmnist.save_train_history('./my_train_hist')

    Example of plotting the accuracy of a saved training history:
        fmnist = FashionMNISTClassifier()
        fmnist.load_train_history('./my_train_hist')
        fmnist.plot_hist(show=true)

    Example of picking a pre-trained model and continue training:
        fmnist = FashionMNISTClassifier()
        fmnist.load_dataset()
        fmnist.load_model('./my_model.h5')
        fmnist.train(64, 5, 0.2)
        fmnist.load_model('./finetuned_model.h5')
        fmnist.save_train_history('./other_train_hist') # saves a different file
    """

    def __init__(self):
        """Initialize the model, train_hist and train_data variables"""
        self.model = None
        self.train_hist = None
        self.train_data = None

    def load_dataset(self):
        """
        The first time the Fashion MNIST dataset is loaded, it will be fetched
        from https://storage.googleapis.com/tensorflow/tf-keras-datasets/
        This is done automatically.

        To change the size of the dataset (for testing purposes, for example),
        simply override the `self.train_data` variable:

        dataset = fmnist.train_data
        fmnist.train_data = (dataset[0][:100], dataset[0][:100])
        """
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        assert x_train.shape == (60000, 28, 28)
        assert y_train.shape == (60000,)
        assert x_test.shape == (10000, 28, 28)
        assert y_test.shape == (10000,)

        self.train_data = (x_train, y_train)

    def build_model(self):
        """
        Build a Sequential model that receives a 28x28 grayscale image
        with pixel values [0, 255] and outputs a Dense layer with 10 units
        corresponding to the classes.

        The first layer scales the input to a float from [0,1]
        Then there are two Convolutional layers followed by a MaxPool.
        The output of the CNN stage is flattened and processed through two
        fully connected layers with ReLU.

        In total, this model contains 507,530 trainable parameters.

        To build a different model simply override the `.model` variable.
        """
        self.model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(28, 28, 1)),
            keras.layers.experimental.preprocessing.Rescaling(1. / 255),
            keras.layers.Conv2D(16, (3, 3), activation='relu'),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((3, 3), strides=2),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

    def show_summary(self):
        """
        Prints the layers that build the model, their output shape,
        and the total number of parameters.
        """
        print(self.model.summary())

    def compile(self):
        """
        Compiles the model using the Sparse Categorical Crossentropy loss
        function and the ADAM optimizer. It measures with accuracy.

        To use a different compile method simply load the model and call
        `.model.compile` with the desired parameters.
        """
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

    def train(self, batch_size, epochs, val_split):
        """
        Train the model using the previously loaded dataset, and receives
        the parameters batch_size, epochs and va_split. The training histoty
        will be stored in the `.train_hist` variable.
        """
        fit_hist = self.model.fit(
            self.train_data[0],
            self.train_data[1],
            batch_size=batch_size,
            epochs=epochs,
            validation_split=val_split
        )
        self.train_hist = fit_hist.history

    def plot_hist(self, show=False, fig_path=None):
        """
        Plot the `accuracy` and `val_accuracy` of the generated training
        history, and either shows it in the current process or saves it.
        """
        x_points = np.arange(1, len(self.train_hist['accuracy']) + 1)

        plt.plot(x_points, self.train_hist['accuracy'], label='Train')
        plt.plot(x_points, self.train_hist['val_accuracy'], label='Validation')

        plt.title('Training accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

        if fig_path:
            plt.savefig(fig_path)

        if show:
            plt.show()

    def save_model(self, path):
        """Save the model in the given path"""
        self.model.save(path)

    def load_model(self, path):
        """Load the model from the given path"""
        self.model = keras.models.load_model(path)

    def save_train_history(self, path):
        """Save the training history in the given path"""
        with open(path, 'wb') as file_pi:
            pickle.dump(self.train_hist, file_pi)

    def load_train_history(self, path):
        """Load the training history from the given path"""
        with open(path, 'rb') as file_pi:
            self.train_hist = pickle.load(file_pi)

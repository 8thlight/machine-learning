# Machine Learning packages
import tensorflow.keras as keras
from tensorflow.keras.datasets import fashion_mnist

# Common packages in scientific computing
import matplotlib.pyplot as plt
import numpy as np

# Support packages
import pickle


class FashionMNISTClassifier:
    def __init__(self, model_path, train_history_path):
        self.model_path = model_path
        self.train_history_path = train_history_path
        self.model = None
        self.train_hist = None
        self.train_data = None

    def load_dataset(self):
        # The first time the Fashion MNIST dataset is loaded, it will be fetched
        # from https://storage.googleapis.com/tensorflow/tf-keras-datasets/
        # This is done automatically.
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        assert x_train.shape == (60000, 28, 28)
        assert y_train.shape == (60000,)
        assert x_test.shape == (10000, 28, 28)
        assert y_test.shape == (10000,)

        self.train_data = (x_train, y_train)

    def build_model(self):
        self.model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(28, 28, 1)),
            keras.layers.experimental.preprocessing.Rescaling(1./255),
            keras.layers.Conv2D(16, (3, 3), activation='relu'),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((3, 3), strides=2),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

    def show_summary(self):
        print(self.model.summary())

    def compile(self):
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

    def train(self, batch_size, epochs, val_split):
        fit_hist = self.model.fit(
            self.train_data[0],
            self.train_data[1],
            batch_size=batch_size,
            epochs=epochs,
            validation_split=val_split
        )
        self.train_hist = fit_hist.history

    def plot_hist(self, show_plot, save_plot):
        x = np.arange(1, len(self.train_hist['accuracy'])+1)

        plt.plot(x, self.train_hist['accuracy'], label='Train')
        plt.plot(x, self.train_hist['val_accuracy'], label='Validation')

        plt.title('Training accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

        if save_plot:
            plt.savefig(save_plot)

        if show_plot:
            plt.show()

    def save_model(self):
        self.model.save(self.model_path)

    def load_model(self):
        self.model = keras.models.load_model(self.model_path)

    def save_train_history(self):
        with open(self.train_history_path, 'wb') as file_pi:
            pickle.dump(self.train_hist, file_pi)

    def load_train_history(self):
        with open(self.train_history_path, 'rb') as file_pi:
            self.train_hist = pickle.load(file_pi)

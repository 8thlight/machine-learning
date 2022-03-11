import tensorflow
import tensorflow.compat.v1 as tf

from .fashion_mnist_classifier import FashionMNISTClassifier

tensorflow.random.set_seed(123)

def test_full_cycle():
    classifier = FashionMNISTClassifier("./model.h5", "./hist")
    classifier.build_model()
    classifier.load_dataset()
    classifier.train_data = ( # just use 300 data samples for unit testing
        classifier.train_data[0][:300],
        classifier.train_data[1][:300]
    )

    classifier.compile()
    classifier.train(128, 2, 0.1)

    assert classifier.model # The model is accessible
    assert classifier.train_hist # The training history is accessible
    assert classifier.train_hist.get('accuracy') # The accuracy was recorded

    accuracy = classifier.train_hist['accuracy']

    assert len(accuracy) == 2 # There is one measurement for every epoch
    assert accuracy[0] < accuracy [1] # The second epoch is an improvement

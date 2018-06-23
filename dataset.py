"""
Provides basic functionality for creating a training data set
as well as a validation set. Additionally provides logic
for batching the images.
"""
import glob
import os
import cv2
from sklearn.utils import shuffle
import numpy as np

def load_train(train_path, image_size, classes):
    images = []
    labels = []

    print("Reading training images")
    for class_ in classes:
        index = classes.index(class_)
        print("Reading for class {} (index: {})".format(class_, index))
        class_dir = os.path.join(train_path, class_, "*g")
        class_files = glob.glob(class_dir)
        for file in class_files:
            image = cv2.imread(file)
            image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


class DataSet(object):
    def __init__(self, images, labels):
        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # After each epoch we update this
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return (
            self._images[start:end],
            self._labels[start:end],
        )


def read_train_sets(train_path, image_size, classes, validation_size):
    images, labels = load_train(train_path, image_size, classes)
    images, labels = shuffle(images, labels)

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]

    training = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)

    return training, validation

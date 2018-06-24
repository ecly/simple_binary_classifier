"""
Used for predicting whether an image is of a given class.
Takes the path of the input to classify as argument on the command line.
"""
import os
import glob
import cv2
import numpy as np
import tensorflow as tf

# Disable warning regarding AVX/FMA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Parameters aligned with those of train.py
IMAGE_SIZE = 256
NUM_CHANNELS = 3
TRAIN_PATH = os.path.join(os.getcwd(), "training_data")
TEST_PATH = os.path.join(os.getcwd(), "testing_data")
CLASSES = os.listdir(TRAIN_PATH)
NUM_CLASSES = len(CLASSES)


def prepare_image(image_path):
    """
    Transforms an image at the given path into the form
    expected as input for the graph.
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), 0, 0, cv2.INTER_LINEAR)
    image_input = [image]
    image_input = np.array(image_input, dtype=np.uint8)
    image_input = image_input.astype("float32")
    image_input = np.multiply(image_input, 1.0 / 255.0)
    # The input to the network is of shape [None IMAGE_SIZE IMAGE_SIZE NUM_CHANNELS]. Hence we reshape.
    return image_input.reshape(1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)


def prepare_graph(session):
    """
    Prepares the graph from the saved model/checkpoint.
    Returns
        TODO
    """
    # Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph("anticheat_model.meta")
    # Load the weights saved using the restore method.
    saver.restore(session, tf.train.latest_checkpoint("./"))
    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")

    # Return the placeholders of the graph such that they can be filled, during testing.
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    return x, y_pred, y_true


def print_results(right, wrong, class_="Total"):
    right_amount = len(right)
    wrong_amount = len(wrong)
    total_amount = right_amount + wrong_amount
    print("\nResults for %s" % class_)
    print("Amount evaluated for class: %d" % (total_amount))
    print("Correct percentage: %f%%" % (right_amount / total_amount))


def evaluate():
    """
    Evaluates the images at TEST_PATH, printing out the results
    for the each class as well as the total class.

    The classes in TEST_PATH are expected to match those at
    TRAIN_PATH.
    """
    classes = os.listdir(TEST_PATH)
    total_wrong = []
    total_right = []
    session = tf.Session()
    x, y_pred, y_true = prepare_graph(session)
    y_test = np.zeros((1, NUM_CLASSES))

    for idx, class_ in enumerate(classes):
        right = []
        wrong = []
        class_dir = os.path.join(TEST_PATH, class_, "*g")
        class_files = glob.glob(class_dir)
        print("Evaluating %d files for class: %s" % (len(class_files), class_))

        for img in class_files:
            img_path = os.path.join(class_dir, img)
            x_batch = prepare_image(img_path)
            # Creating the feed_dict that is required to be fed to calculate y_pred
            feed_dict = {x: x_batch, y_true: y_test}
            [res] = session.run(y_pred, feed_dict=feed_dict)
            probability = res[idx]
            if probability == max(res):
                right.append({img_path, probability})
            else:
                wrong.append({img_path, probability})

        print_results(right, wrong, class_)
        total_right.extend(right)
        total_wrong.extend(wrong)

    print_results(total_right, total_wrong)
    session.close()


if __name__ == "__main__":
    evaluate()

import os
import sys
import cv2
import numpy as np
import tensorflow as tf

# Disable warning regarding AVX/FMA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Parameters aligned with those of train.py
IMAGE_SIZE = 128
NUM_CHANNELS = 3
TRAIN_PATH = os.path.join(os.getcwd(), "training_data")
CLASSES = os.listdir(TRAIN_PATH)
NUM_CLASSES = len(CLASSES)

def predict(image_path):
    images = []
    # Reading the image using OpenCV
    image = cv2.imread(image_path)
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), 0, 0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype("float32")
    images = np.multiply(images, 1.0 / 255.0)
    # The input to the network is of shape [None IMAGE_SIZE IMAGE_SIZE NUM_CHANNELS]. Hence we reshape.
    x_batch = images.reshape(1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    print(x_batch)

    ## Let us restore the saved model
    with tf.Session() as session:
        # Step-1: Recreate the network graph. At this step only graph is created.
        saver = tf.train.import_meta_graph("cheat_not_cheat_model.meta")
        # Step-2: Now let's load the weights saved using the restore method.
        saver.restore(session, tf.train.latest_checkpoint("./"))
        # Accessing the default graph which we have restored
        graph = tf.get_default_graph()

        # Now, let's get hold of the op that we can be processed to get the output.
        # In the original network y_pred is the tensor that is the prediction of the network
        y_pred = graph.get_tensor_by_name("y_pred:0")

        ## Let's feed the images to the input placeholders
        x = graph.get_tensor_by_name("x:0")
        y_true = graph.get_tensor_by_name("y_true:0")
        y_test_images = np.zeros((1, NUM_CLASSES))


        ### Creating the feed_dict that is required to be fed to calculate y_pred
        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        result = session.run(y_pred, feed_dict=feed_dict_testing)
        # result is of this format [probability_of_cheat, probability_of_not_cheat]
        return result

if __name__ == "__main__":
    [[CHEAT, NOT_CHEAT]] = predict(os.path.join(os.getcwd(), sys.argv[1]))
    if CHEAT > NOT_CHEAT:
        print("Cheat with %f confidence"%CHEAT)
    else:
        print("Not-cheat with %f confidence"%NOT_CHEAT)

"""
http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
"""
import os
import tensorflow as tf
from numpy.random import seed
import dataset

# Disable warning regarding AVX/FMA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Add Seed so that random initialization is consistent
seed(1)
tf.set_random_seed(2)

TRAIN_PATH = os.path.join(os.getcwd(), "training_data")
MODEL_NAME = "cheat_not_cheat_model"

# The classes we'll be using, in the form of the folders contained in TRAIN_PATH
CLASSES = os.listdir(TRAIN_PATH)
NUM_CLASSES = len(CLASSES)

BATCH_SIZE = 32


# 20% of the data will automatically be used for validation
VALIDATION_SIZE = 0.2
IMG_SIZE = 128
NUM_CHANNELS = 3


# Network graph params
FILTER_SIZE_CONV1 = 3
NUM_FILTERS_CONV1 = 32

FILTER_SIZE_CONV2 = 3
NUM_FILTERS_CONV2 = 32

FILTER_SIZE_CONV3 = 3
NUM_FILTERS_CONV3 = 64

FC_LAYER_SIZE = 128


def create_weights(shape):
    """ Creates the initial weights of the network """
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    """ Creates the initial biases of the network """
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(
    input_, num_input_channels, conv_filter_size, num_filters
):
    """
    Helper function for creating a convolutional layer for the network.
    Args:
    input_: the output(activation) from the previous layer. A 4D tensor.
    num_input_channels: number of channels, eg. RGB images 3
    conv_filter_size: the size of the convolutional filter
    num_filters: the amount of filters
    """

    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(
        shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters]
    )
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(
        input=input_, filter=weights, strides=[1, 1, 1, 1], padding="SAME"
    )

    layer += biases

    ## We shall be using max-pooling.
    layer = tf.nn.max_pool(
        value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
    )
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):
    """
    Helper function for creating a flattening layer
    based on the previous layer in the network.
    """
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input_, num_inputs, num_outputs, use_relu=True):
    """
    Helper function for creating a fully connected layer.
    Args:
    input: the output(activation) from the previous layer. A 4D tensor.
    num_input: the size of the input layer's output
    num_outputs: the size of the output of the fc layer
    use_relu:
    https://stackoverflow.com/questions/43504248/what-does-relu-stand-for-in-tf-nn-relu
    """

    # Define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.
    # Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input_, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


def prepare_data():
    """
    Prepares the data in the training path.
    Here we expect an amount of folders equal to the amount of classes,
    where each folder will contain only images of that class.
    """
    data = dataset.read_train_sets(
        TRAIN_PATH, IMG_SIZE, CLASSES, validation_size=VALIDATION_SIZE
    )

    print("Complete reading input data. Will Now print a snippet of it")
    print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
    print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

    return data


def prepare_graph():
    """
    Prepares the grpah that we'll use for training
    This consists of 3 convolutional layers, a flattening layer,
    a fully connected layer, and a fully connected output.
    """
    x = tf.placeholder(
        tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, NUM_CHANNELS], name="x"
    )

    # Labels
    y_true = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name="y_true")
    y_true_cls = tf.argmax(y_true, axis=1)

    layer_conv1 = create_convolutional_layer(
        input_=x,
        num_input_channels=NUM_CHANNELS,
        conv_filter_size=FILTER_SIZE_CONV1,
        num_filters=NUM_FILTERS_CONV1,
    )

    layer_conv2 = create_convolutional_layer(
        input_=layer_conv1,
        num_input_channels=NUM_FILTERS_CONV1,
        conv_filter_size=FILTER_SIZE_CONV2,
        num_filters=NUM_FILTERS_CONV2,
    )

    layer_conv3 = create_convolutional_layer(
        input_=layer_conv2,
        num_input_channels=NUM_FILTERS_CONV2,
        conv_filter_size=FILTER_SIZE_CONV3,
        num_filters=NUM_FILTERS_CONV3,
    )

    layer_flat = create_flatten_layer(layer_conv3)

    layer_fc1 = create_fc_layer(
        input_=layer_flat,
        num_inputs=layer_flat.get_shape()[1:4].num_elements(),
        num_outputs=FC_LAYER_SIZE,
        use_relu=True,
    )

    layer_fc2 = create_fc_layer(
        input_=layer_fc1,
        num_inputs=FC_LAYER_SIZE,
        num_outputs=NUM_CLASSES,
        use_relu=False,
    )

    y_pred = tf.nn.softmax(layer_fc2, name="y_pred")

    y_pred_cls = tf.argmax(y_pred, axis=1)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=layer_fc2, labels=y_true
    )
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return optimizer, accuracy, cost, x, y_true


def show_progress(
    session, accuracy, epoch, feed_dict_train, feed_dict_validate, val_loss
):
    """ Prints the progress of the training """
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = (
        "Training Epoch {0} - Training Acc.: {1:>6.1%}, "
        + "Validation Acc.: {2:>6.1%},  Validation Loss: {3:.3f}"
    )
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


def train(num_iteration):
    """ Trains the network using the given amount of the iterations """
    data = prepare_data()
    optimizer, accuracy, cost, x, y_true = prepare_graph()
    total_iterations = 0
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for i in range(total_iterations, total_iterations + num_iteration):

            x_batch, y_true_batch, _, _ = data.train.next_batch(BATCH_SIZE)
            x_valid_batch, y_valid_batch, _, _ = data.valid.next_batch(BATCH_SIZE)

            feed_dict_tr = {x: x_batch, y_true: y_true_batch}
            feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}

            session.run(optimizer, feed_dict=feed_dict_tr)

            if i % int(data.train.num_examples / BATCH_SIZE) == 0:
                val_loss = session.run(cost, feed_dict=feed_dict_val)
                epoch = int(i / int(data.train.num_examples / BATCH_SIZE))

                show_progress(
                    session, accuracy, epoch, feed_dict_tr, feed_dict_val, val_loss
                )
                saver.save(session, os.path.join(os.getcwd(), MODEL_NAME))

        total_iterations += num_iteration


if __name__ == "__main__":
    train(num_iteration=3000)

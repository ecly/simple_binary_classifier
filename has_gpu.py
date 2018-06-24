import os
import tensorflow as tf

# Disable warning regarding AVX/FMA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

with tf.device("/gpu:0"):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name="a")
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name="b")
    c = tf.matmul(a, b)

with tf.Session() as sess:
    try:
        sess.run(c)
        print("GPU acceleration working")
    except:
        print("GPU acceleration not working")

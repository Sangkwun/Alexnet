import tensorflow as tf


def alexnet(width=224, height=224, classes=10, batchnorm = True):
    # Input shape is 224, 224, 3
    X = tf.placeholder(tf.float32, [None, width, height, 3])

    # Output is for Pascal VOC 10 classes
    Y = tf.placeholder(tf.float32, [None, 10])
    is_training = tf.placeholder(tf.bool)

    # First convolutional layer
    L1 = tf.layers.conv2d(X, filters=96, kernel_size=[11, 11], strides=(4, 4), activation=tf.nn.relu, name='conv1') # 55, 55, 96
    if batchnorm:
        L1 = tf.layers.batch_normalization(L1, 2, 1e-04, 0.75, name='norm1')
    L1 = tf.layers.max_pooling2d(L1, [3, 3], [2, 2], padding='valid', name='poo1') # 27, 27, 96

    # Second convolutional layer
    L2 = tf.layers.conv2d(L1, filters=256, kernel_size=[5, 5], strides=(1, 1), activation=tf.nn.relu, name='conv2')
    if batchnorm:
        L2 = tf.layers.batch_normalization(L2, 2, 1e-04, 0.75, name='norm2')
    L2 = tf.layers.max_pooling2d(L2, [3, 3], [2, 2], padding='valid', name='poo1') # 27, 27, 96




alexnet()
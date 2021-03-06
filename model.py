import tensorflow as tf


def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.summary.scalar('sttdev/' + name, stddev)
    tf.summary.scalar('max/' + name, tf.reduce_max(var))
    tf.summary.scalar('min/' + name, tf.reduce_min(var))
    tf.summary.histogram(name, var)

def alexnet(width=227, height=227, classes=5, dropout=0.5, batchnorm = True):

    """
        (?, 227, 227, 3)
        (?, 27, 27, 96)
        (?, 13, 13, 256)
        (?, 13, 13, 384)
        (?, 13, 13, 384)
        (?, 6, 6, 256)
        (?, 4096)
        (?, 4096)
        (?, 5)
    """

    # Input shape is 227, 227, 3
    X = tf.placeholder(tf.float32, [None, width, height, 3])
    print(X.shape)

    # Output is for Pascal VOC 10 classes
    Y = tf.placeholder(tf.float32, [None, classes])
    is_training = tf.placeholder(tf.bool)

    # First convolutional layer
    L1 = tf.layers.conv2d(X, filters=96, kernel_size=[11, 11], strides=(4, 4), activation=tf.nn.relu, name='conv1') # 55, 55, 96
    if batchnorm:
        L1 = tf.layers.batch_normalization(L1, axis=2, momentum=1e-04, epsilon=0.75, name='norm1')
    L1 = tf.layers.max_pooling2d(L1, pool_size=[3, 3], strides=[2, 2], padding='valid', name='poo1') # 27, 27, 96
    print(L1.shape)

    # Second convolutional layer
    L2 = tf.layers.conv2d(L1, filters=256, kernel_size=[5, 5], strides=(1, 1), padding='same', activation=tf.nn.relu, name='conv2') # 27, 27, 256
    if batchnorm:
        L2 = tf.layers.batch_normalization(L2, axis=2, momentum=1e-04, epsilon=0.75, name='norm2')
    L2 = tf.layers.max_pooling2d(L2, pool_size=[3, 3], strides=[2, 2], padding='valid', name='poo2') # 13, 13, 256
    print(L2.shape)

    # Third convolutional layer
    L3 = tf.layers.conv2d(L2, filters=384, kernel_size=[3, 3], strides=(1, 1), padding='same', activation=tf.nn.relu, name='conv3') # 13, 13, 384
    print(L3.shape)

    # Fourth convolutional layer
    L4 = tf.layers.conv2d(L3, filters=384, kernel_size=[3, 3], strides=(1, 1), padding='same', activation=tf.nn.relu, name='conv4') # 13, 13, 384
    print(L4.shape)

    # Fifth convolutional layer
    L5 = tf.layers.conv2d(L4, filters=256, kernel_size=[3, 3], strides=(1, 1), padding='same', activation=tf.nn.relu, name='conv5') # 13, 13, 256
    L5 = tf.layers.max_pooling2d(L5, pool_size=[3, 3], strides=[2, 2], padding='valid', name='pool5') # 6, 6, 256
    print(L5.shape)

    # Flatten layer
    L6 = tf.contrib.layers.flatten(L5) # 9216
    L6 = tf.layers.dense(L6, 4096, activation=tf.nn.relu, name='dense6') # 4096
    L6 = tf.layers.dropout(L6, dropout, name='drop6')
    print(L6.shape)
    
    L7 = tf.layers.dense(L6, 4096, activation=tf.nn.relu, name='dense7') # 4096
    L7 = tf.layers.dropout(L7, dropout, name='drop7')
    print(L7.shape)

    model = tf.layers.dense(L7, classes, activation=None, name='dense8') # 5

    print(model.shape)

    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
        variable_summaries(cost, '/cost')

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        
        variable_summaries(accuracy, '/accuracy')

    merged = tf.summary.merge_all()


    return X, Y, is_training, cost, optimizer, accuracy, merged, model

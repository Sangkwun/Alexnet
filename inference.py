import tensorflow as tf
import numpy as np

from model import alexnet
from data import image_to_array

width=227
height=227

meta_path = 'weights/-1335.meta'
ckpt_path = 'weights/'
image_path = 'resources/flowers/rose/110472418_87b6a3aa98_m.jpg'
init = tf.global_variables_initializer()

classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

X, Y, is_training, cost, optimizer, accuracy, merged, model = alexnet(dropout=1)

arr = image_to_array(image_path)
arr = np.stack([arr], axis=0)

feed_dict = {
    X: arr,
    is_training: False
}

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
    result = sess.run(model, feed_dict)
    index = np.argmax(result)
    print(result)
    print(index)

    print("The class of image is {}".format(classes[index]))
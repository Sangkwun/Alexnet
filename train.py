import tensorflow as tf

from model import alexnet
from data import read_dataset, image_to_array, path_to_4dtensor

# Setting for learning
batch_size = 200
iteration = 20

X, Y, is_training, cost, optimizer = alexnet()

# Initialize valuables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Read dataset.
df = read_dataset()
labels = df[['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']]

for i in range(iteration):
    x_input = path_to_4dtensor(paths=df['path'], batch_size=batch_size, num_iter=i)
    label = labels[batch_size * iteration: batch_size * (iteration + 1)]

    _, cost_val = sess.run([optimizer, cost], feed_dict={X: x_input, Y: label, is_training: True})

    print('Epoch:', '%04d' % (i + 1), 'Avg. cost =', '{:.4f}'.format(cost_val))
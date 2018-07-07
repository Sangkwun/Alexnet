import tensorflow as tf

from model import alexnet
from data import read_dataset, image_to_array, path_to_4dtensor


"""
    To.do
    - Apply tensorboard callback
    
"""
# Dataset Path
train_path = 'resources/traing_dataset.csv'
valid_path = 'resources/valid_dataset.csv'
test_path =  'resources/test_dataset.csv'


# Setting for learning
batch_size = 30
iteration = 1
epochs = 100

X, Y, is_training, cost, optimizer, accuracy, is_correct = alexnet()

# Initialize valuables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Read dataset.
classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
train = read_dataset(train_path)
valid = read_dataset(valid_path)
tran_labels = train[classes]
valid_labels = valid[classes]

for e in range(epochs):
    for i in range(iteration):
        # Training
        x_input = path_to_4dtensor(paths=train['path'], batch_size=batch_size, num_iter=i)
        print(batch_size * i, batch_size * (i + 1))
        label = tran_labels[batch_size * i: batch_size * (i + 1)]
        #print(label)
        acc, _, cost_val, correct = sess.run([accuracy, optimizer, cost, is_correct], feed_dict={X: x_input, Y: label, is_training: True})

        print('Training Epoch:', '%04d' % (i + 1), 'Avg. acc =', '{:.4f}'.format(acc), 'Avg. cost =', '{:.4f}'.format(cost_val))
        #print(correct)
        # Validation check
        x_input = path_to_4dtensor(paths=valid['path'], batch_size=20, num_iter=1)
        label = valid_labels[0:20]

        acc, cost_val = sess.run([accuracy, cost], feed_dict={X: x_input, Y: label, is_training: False})
        print('Validation Epoch:', '%04d' % (i + 1), 'Avg. acc =', '{:.4f}'.format(acc), 'Avg. cost =', '{:.4f}'.format(cost_val))
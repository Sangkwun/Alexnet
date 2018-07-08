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
batch_size = 100
iteration = 10
epochs = 10
valid_size = 5

X, Y, is_training, cost, optimizer, accuracy, merged = alexnet()

# Read dataset.
classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
train = read_dataset(train_path)
valid = read_dataset(valid_path)
tran_labels = train[classes]
valid_labels = valid[classes]

sess = tf.Session()

# For tensorboard
logdir='log/'
train_writer = tf.summary.FileWriter(logdir + '/train', sess.graph)
valid_writer = tf.summary.FileWriter(logdir + '/valid')

# Initialize valuables
init = tf.global_variables_initializer()
sess.run(init)

# For saving model
saver = tf.train.Saver()
saver.save(sess, 'weights/flower/')

min_cost = None

for e in range(epochs):
    for i in range(iteration):
        # Training
        x_input = path_to_4dtensor(paths=train['path'], batch_size=batch_size, num_iter=i)
        print(batch_size * i, batch_size * (i + 1))
        label = tran_labels[batch_size * i: batch_size * (i + 1)]
        #print(label)
        acc, _, cost_val, summary = sess.run([accuracy, optimizer, cost, merged], feed_dict={X: x_input, Y: label, is_training: True})
        
        train_writer.add_summary(summary, i + (e * iteration))

        print('Training Epoch:', '%04d' % (i + 1), 'Avg. acc =', '{:.4f}'.format(acc), 'Avg. cost =', '{:.4f}'.format(cost_val))
        #print(correct)
        # Validation check
        x_input = path_to_4dtensor(paths=valid['path'], batch_size=valid_size, num_iter=1)
        label = valid_labels[0:valid_size]

        acc, cost_val, summary = sess.run([accuracy, cost, merged], feed_dict={X: x_input, Y: label, is_training: False})
        print('Validation Epoch:', '%04d' % (i + 1), 'Avg. acc =', '{:.4f}'.format(acc), 'Avg. cost =', '{:.4f}'.format(cost_val))
        valid_writer.add_summary(summary, i + (e * iteration))

        if cost_val is None and cost_val < min_cost:
            min_cost = cost_val
            saver.save(sess, "my_test_model", global_step=i + (e * iteration))

            
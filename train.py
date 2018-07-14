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

X, Y, is_training, cost, optimizer, accuracy, merged, model = alexnet(dropout=1)

# Read dataset.
classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
train = read_dataset(train_path)
valid = read_dataset(valid_path)
tran_labels = train[classes]
valid_labels = valid[classes]

# Setting for learning
batch_size = 100
iteration = int(len(train)/100)
epochs = 100
valid_size = 10

print("batch_size: {}, iteration: {}, valid_size: {}".format(batch_size, iteration, valid_size))

sess = tf.Session()

# For tensorboard
logdir='logdir/'
train_writer = tf.summary.FileWriter(logdir + '/train_3', sess.graph)
valid_writer = tf.summary.FileWriter(logdir + '/valid_3')

# Initialize valuables
init = tf.global_variables_initializer()
sess.run(init)

# For saving model
saver = tf.train.Saver()
saver.save(sess, 'weights/')

min_cost = None

for e in range(epochs):
    for i in range(iteration):
        # Training
        x_input = path_to_4dtensor(paths=train['path'], batch_size=batch_size, num_iter=i)
        label = tran_labels[batch_size * i: batch_size * (i + 1)]

        acc, _, cost_val, summary = sess.run([accuracy, optimizer, cost, merged], feed_dict={X: x_input, Y: label, is_training: True})
        
        train_writer.add_summary(summary, i + (e * iteration))

        print("epoch:{} iteration:{} on {} images".format(e, i, batch_size), 'Avg. acc =', '{:.4f}'.format(acc), 'Avg. cost =', '{:.4f}'.format(cost_val))
        # Validation check
        x_input = path_to_4dtensor(paths=valid['path'], batch_size=valid_size, num_iter=i)
        label = valid_labels[valid_size * i: valid_size * (i + 1)]

        acc, cost_val, summary = sess.run([accuracy, cost, merged], feed_dict={X: x_input, Y: label, is_training: False})
        print("epoch:{} iteration:{} on {} images".format(e, i, valid_size), 'Avg. acc =', '{:.4f}'.format(acc), 'Avg. cost =', '{:.4f}'.format(cost_val))
        valid_writer.add_summary(summary, i + (e * iteration))
        if cost_val < 1:
            if min_cost is None:
                min_cost = cost_val
                saver.save(sess, "weights/", global_step=i + (e * iteration))
            else:
                if cost_val < min_cost:
                    min_cost = cost_val
                    saver.save(sess, "weights/", global_step=i + (e * iteration))


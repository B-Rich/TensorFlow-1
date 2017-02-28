import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
print(xy)
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

x = tf.placeholder("float", [None, 3])
y = tf.placeholder("float", [None, 3])

w = tf.Variable(tf.zeros([3, 3]))

hypothesis = tf.nn.softmax(tf.matmul(x, w))

lr = 0.001

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(2001):
        sess.run(optimizer, feed_dict={x:x_data, y:y_data})
        if step & 20 == 0:
            print(step, end="\n\n")
            print(sess.run(cost, feed_dict={x:x_data, y:y_data}))
            print(sess.run(w))
            print()


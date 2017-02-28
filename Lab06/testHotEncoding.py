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

    for step in range(4001):
        sess.run(optimizer, feed_dict={x:x_data, y:y_data})
        if step & 20 == 0:
            print(step)
            print(sess.run(cost, feed_dict={x:x_data, y:y_data}))
            print(sess.run(w))
            print()

    print('--------------------')

    a = sess.run(hypothesis, feed_dict={x: [[1, 11, 7]]})
    print(a, sess.run(tf.argmax(a, 1)))

    b = sess.run(hypothesis, feed_dict={x: [[1, 3, 4]]})
    print(a, sess.run(tf.argmax(b, 1)))

    c = sess.run(hypothesis, feed_dict={x: [[1, 1, 0]]})
    print(a, sess.run(tf.argmax(c, 1)))

    all = sess.run(hypothesis, feed_dict={x: [[1, 11, 7], [1, 3, 4], [1, 1, 0]]})
    print(all, sess.run(tf.argmax(all, 1)))

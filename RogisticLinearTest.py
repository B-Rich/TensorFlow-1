import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack = True, dtype ='float32')
print(xy)

x_data = xy[0:-1]
y_data = xy[-1]

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

h = tf.matmul(w, x)
hypothesis = tf.div(1., 1.+tf.exp(-h))

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={x:x_data, y:y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={x: x_data, y: y_data}), sess.run(w))

print('---------------------------')

print(sess.run(hypothesis, feed_dict={x: [[1], [2], [2]]}) > 0.5)
print(sess.run(hypothesis, feed_dict={x: [[1], [5], [5]]}) > 0.5)
print(sess.run(hypothesis, feed_dict={x: [[1, 1], [4, 3], [3, 5]]}) > 0.5)

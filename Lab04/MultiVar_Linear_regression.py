import tensorflow as tf


x1_data = [1, 0, 3, 0, 5]
x2_data = [0, 2, 0, 4, 0]
y_data = [1, 2, 3, 4, 5]

W1 = tf.Variable(tf.random_uniform([1], -10.0, 10.0))
W2 = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

b = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)

y = tf.placeholder(tf.float32)

hypothesis = W1 * x1 + W2 * x2 + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={x1:x1_data, x2:x2_data, y:y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={x1:x1_data, x2:x2_data, y:y_data}), sess.run(W1), sess.run(W2), sess.run(b))

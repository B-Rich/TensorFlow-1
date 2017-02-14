import tensorflow as tf


x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

hypothesis = W * x

cost = tf.reduce_mean(tf.square(hypothesis - y))

lr = 0.1
descent = W - tf.mul(lr, tf.reduce_mean(tf.mul((tf.mul(W, x) - y), x)))
train = W.assign(descent)


init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train, feed_dict={x:x_data, y:y_data})

    print(step, sess.run(cost, feed_dict={x:x_data, y:y_data}), sess.run(W), )

print(sess.run(hypothesis, feed_dict={x:5}))

print(sess.run(hypothesis, feed_dict={x:2.5}))

print(sess.run(hypothesis, feed_dict={x:500}))
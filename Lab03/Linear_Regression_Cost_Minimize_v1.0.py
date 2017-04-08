import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)

x = [1, 2, 3]
y = [2, 3, 4]

w = tf.placeholder(tf.float32)
hypo = w * x

cost = tf.reduce_mean(tf.square(hypo - y))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

w_history = []
cost_history = []

for i in range(-30, 50):
    feed_w = i * 0.1
    curr_cost, curr_w = sess.run([cost, w], feed_dict={w: feed_w})
    w_history.append(curr_w)
    cost_history.append(curr_cost)

plt.plot(w_history, cost_history)
plt.show()
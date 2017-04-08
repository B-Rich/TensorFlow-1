import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1], name='Weight'))
b = tf.Variable(tf.random_normal([1], name='Bias'))

hypo = (w * x) - b

cost = tf.reduce_mean(tf.square(hypo - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(4001):
    _, w_, b_, cost_ =sess.run([train, w, b, cost], feed_dict={x : [1, 2, 3, 4], y : [2, 3, 4, 5]})
    if step % 20 == 0:
        print(step, cost_, w_, b_, _)

print(sess.run(hypo, feed_dict={x:[4]}))
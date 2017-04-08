import tensorflow as tf
tf.set_random_seed(777)

x = [1, 2, 3]
y = [2, 3, 4]

w = tf.Variable(tf.random_normal([1]), name='Weight')
b = tf.Variable(tf.random_normal([1]), name='Bias')

hypo = (x * w) + b

cost = tf.reduce_mean(tf.square(hypo - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost))
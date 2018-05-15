import tensorflow as tf
import numpy as np

with open('./Random_Sampling_full.csv') as f:
    lines = (line for line in f if not line.startswith('#'))
    data = np.loadtxt(lines, delimiter=',', unpack=True, dtype='float32')

input_data = np.transpose(data[0:2])
actual_data = np.transpose(data[2:])
rssi_data = np.transpose(data[1:-1])
tx_data = np.transpose(data[0:1])

coef1 = 0.42093
coef2 = 6.9476
coef3 = 0.54992


X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
RSSI = tf.placeholder(tf.float32)
TX = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 4]))
W2 = tf.Variable(tf.random_uniform([4, 3]))
W3 = tf.Variable(tf.random_uniform([3, 1]))
W3 = tf.Variable(tf.random_uniform([3, 1]))
W3 = tf.Variable(tf.random_uniform([3, 1]))

b1 = tf.Variable(tf.zeros([4]))
b2 = tf.Variable(tf.zeros([3]))
b3 = tf.Variable(tf.zeros([1]))


L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.sigmoid(L1)
L2 = tf.add(tf.matmul(L1, W2), b2)
L2 = tf.nn.relu(L2)
L3 = tf.add(tf.matmul(L2, W3), b3)
L2 = tf.add(tf.matmul(L1, W2), b2)
# L3 = tf.nn.relu(L3)

predicted = L3
# wei = np.asarray(predicted)
# distance = wei[0]*coef1*pow((RSSI*1.00)/TX, wei[1]*coef2) + wei[2]*coef3
# cost = tf.reduce_mean(tf.square(Y-distance))
cost = tf.reduce_mean(tf.square(Y-predicted))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(2111):
    # sess.run(train_op, feed_dict={X: input_data, Y: actual_data, RSSI:rssi_data, TX:tx_data})
    sess.run(train_op, feed_dict={X: input_data, Y: actual_data})
    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: input_data, Y: actual_data}))

print('예측W : ', sess.run(W3, feed_dict={X:input_data}))
print('예측값 : ', sess.run(predicted, feed_dict={X:input_data}))
print('실측값 : ', sess.run(Y, feed_dict={Y:actual_data}))

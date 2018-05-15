import tensorflow as tf
import numpy as np

class model:

    def __init__(self):

        self.session = tf.Session()
        self.coef1 = 0.42093
        self.coef2 = 6.9476
        self.coef3 = 0.54992

        self.input_X = tf.placeholder(tf.float32, [None, 2])
        self.input_Y = tf.placeholder(tf.float32, [None])
        self.input_RSSI = tf.placeholder(tf.float32, shape=[None])
        self.input_TX = tf.placeholder(tf.float32, shape=[None])

        self.distance = self._build_network('main')
        self.loss, self.train_op = self._build_op()

    def _build_network(self, name):
        with tf.variable_scope(name):
            model = tf.layers.dense(self.input_X, 2, activation=tf.sigmoid)
            model = tf.layers.dense(model, 4, activation=tf.sigmoid)
            model = tf.layers.dense(model, 8, activation=tf.sigmoid)
            model = tf.layers.dense(model, 3, activation=tf.sigmoid)
            W = tf.layers.dense(model, 3, activation=tf.sigmoid)
            distance = W[0]*self.coef1*pow((self.input_RSSI*1.00)/self.input_TX, W[1]*self.coef2) + W[2]*self.coef3
            return distance

    def _build_op(self):
        loss = tf.reduce_mean(tf.square(self.input_Y - self.distance))
        train_op = tf.train.AdamOptimizer(1e-6).minimize(loss)

        print('실측거리 : '+str(self.input_Y[0]))
        print('예측거리 : '+str(self.distance[0]))
        return loss, train_op

    def train(self, index):
        with open('./Random_Sampling_full.csv') as f:
            lines = (line for line in f if not line.startswith('#'))
            data = np.loadtxt(lines, delimiter=',', unpack=True, dtype='float32')

        input_data = np.transpose(data[0:2])[index]
        actual_data = np.transpose(data[2:])[index]
        rssi_data = np.transpose(data[1:-1])[index]
        tx_data = np.transpose(data[0:1])[index]

        print(tx_data)

        self.session.run(tf.global_variables_initializer())


        self.session.run(self.train_op,
                    feed_dict={
                        self.input_X: input_data,
                        self.input_Y: actual_data,
                        self.input_RSSI: rssi_data,
                        self.input_TX: tx_data
                    })

    def loop(self):
        terminal = False
        episode = 0
        while not terminal:
            self.train(episode)
            episode += 1


def main(_):
    model().loop()


if __name__ == '__main__':
    tf.app.run()
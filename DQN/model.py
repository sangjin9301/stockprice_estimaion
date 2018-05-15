import tensorflow as tf
import numpy as np
import random
from collections import deque


class DQN:
    # 학습에 사용할 플레이결과를 얼마나 많이 저장해서 사용할지를 정합니다.
    # (플레이결과 = 게임판의 상태 + 액션 + 리워드 + 종료여부)
    REPLAY_MEMORY = 100
    # 학습시 사용/계산할 상태값(정확히는 replay memory)의 갯수를 정합니다.
    BATCH_SIZE = 10
    # 과거의 상태에 대한 가중치를 줄이는 역할을 합니다.
    GAMMA = 0.9
    # 한 번에 볼 총 프레임 수 입니다.
    # 앞의 상태까지 고려하기 위함입니다.
    STATE_LEN = 5

    def __init__(self, session, n_action):
        self.session = session
        self.n_action = n_action
        # 게임 플레이결과를 저장할 메모리
        self.memory = deque()
        # 현재 게임판의 상태
        self.state = None

        self.input_X = tf.placeholder(tf.float32, [None, 40, self.STATE_LEN]) #state => 1 x 40
        self.input_A = tf.placeholder(tf.int64, [None])
        self.input_Y = tf.placeholder(tf.float32, [None])

        self.Q = self._build_network('main')
        self.cost, self.train_op = self._build_op()

        self.target_Q = self._build_network('target')

    def _build_network(self, name):

        with tf.variable_scope(name):
            model = tf.layers.dense(self.input_X, 40, activation=tf.nn.relu)
            model = tf.layers.dense(model, 80, activation=tf.nn.sigmoid)
            model = tf.layers.dense(model, 256, activation=tf.nn.relu)
            model = tf.layers.dense(model, 512, activation=tf.nn.sigmoid)
            model = tf.contrib.layers.flatten(model)
            model = tf.layers.dense(model, 80, activation=tf.nn.relu)

            Q = tf.layers.dense(model, self.n_action, activation=tf.nn.sigmoid)

        return Q

    def _build_op(self):
        one_hot = tf.one_hot(self.input_A, self.n_action, 1.0, 0.0)
        Q_value = tf.reduce_sum(tf.multiply(self.Q, one_hot), axis=1)
        cost = tf.reduce_mean(tf.square(self.input_Y - Q_value))
        train_op = tf.train.AdamOptimizer(1e-6).minimize(cost)

        return cost, train_op

    def update_target_network(self):
        copy_op = []

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))

        self.session.run(copy_op)

    def get_action(self):
        Q_value = self.session.run(self.Q,
                                   feed_dict={self.input_X: [self.state]})

        action = np.argmax(Q_value[0])
        print("Q_value : " + str(Q_value[0]))

        return action

    def init_state(self, state):
        state = [state for _ in range(self.STATE_LEN)]

        self.state = np.stack(state, axis=1)

    def remember(self, state, action, reward, terminal):
        next_state = np.reshape(state, (40, 1))
        next_state = np.append(self.state[:, 1:], next_state, axis=1)

        self.memory.append((self.state, next_state, action, reward, terminal))

        self.state = next_state
        if len(self.memory) > self.REPLAY_MEMORY:
            self.memory.popleft()


    def _sample_memory(self):
        sample_memory = random.sample(self.memory, self.BATCH_SIZE)

        state = [memory[0] for memory in sample_memory]
        next_state = [memory[1] for memory in sample_memory]
        action = [memory[2] for memory in sample_memory]
        reward = [memory[3] for memory in sample_memory]
        terminal = [memory[4] for memory in sample_memory]

        return state, next_state, action, reward, terminal

    def train(self):
        state, next_state, action, reward, terminal = self._sample_memory()


        target_Q_value = self.session.run(self.target_Q,
                                          feed_dict={self.input_X: next_state})

        Y = []
        for i in range(self.BATCH_SIZE):
            if terminal[i]:
                Y.append(reward[i])
            else:
                Y.append(reward[i] + self.GAMMA * np.max(target_Q_value[i]))


        self.session.run(self.train_op,
                         feed_dict={
                             self.input_X: state,
                             self.input_A: action,
                             self.input_Y: Y
                         })
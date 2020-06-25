import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
import numpy as np
import random
import collections

class DQN(keras.Model):
    def __init__(self, input_size, output_size, mem_size):
        super(DQN,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.memory = collections.deque(maxlen=mem_size)
        self.net = keras.Sequential([
            layers.Dense(128, activation = tf.nn.relu),
            layers.Dense(128, activation = tf.nn.relu),
            layers.Dense(64, activation = tf.nn.relu),
            layers.Dense(self.output_size)
        ]
        )
        # ICM structure     back_net:  at + st predict st+1  for_net: st + st+1 predict at
        self.back_net = keras.Sequential([
            layers.Dense(64, activation = tf.nn.relu),
            layers.Dense(self.input_size)
        ]
        )
        self.for_net = keras.Sequential([
            layers.Dense(64, activation = tf.nn.relu),
            layers.Dense(2)
        ]
        )


    def call(self, inputs):
        q_value = self.net(inputs)
        return q_value

    def save_trans(self, transition):
        self.memory.append(transition)

    def choose_action(self, inputs, epsilon):
        inputs = tf.constant(inputs, dtype = tf.float32)
        inputs = tf.expand_dims(inputs, axis = 0)
        q_value = self(inputs)[0]
        # print(q_value)
        coin = np.random.rand()
        if coin > epsilon:
            # print(tf.to_int32(tf.argmax(q_value)))
            # print(int(tf.argmax(q_value)))
            return int(tf.argmax(q_value))
        else:
            return np.random.randint(0, self.output_size)


    def sample_batch(self, batch_size):
        trans_batch = random.sample(self.memory, batch_size)
        s_ls, a_ls, r_ls, s_next_ls, done_flag_ls = [], [], [], [], []
        for trans in trans_batch:
            s, a, r, s_next, done_flag = trans
            s_ls.append(s)
            a_ls.append([a])
            r_ls.append([r])
            s_next_ls.append(s_next)
            done_flag_ls.append([done_flag])
        return tf.constant(s_ls, dtype=tf.float32),\
                tf.constant(a_ls, dtype=tf.int32),\
                tf.constant(r_ls, dtype=tf.float32),\
                tf.constant(s_next_ls, dtype=tf.float32),\
                tf.constant(done_flag_ls, dtype=tf.float32)

    def cal_ICMreward(self, s, s_next, a):
        s = tf.constant(s, dtype=tf.float32)
        a = tf.constant(a, dtype=tf.float32)
        a = tf.expand_dims(a, axis=0)
        inputs = tf.concat([s, a], axis=0)
        inputs = tf.expand_dims(inputs, axis=0)
        s_next_pred = self.back_net(inputs).numpy()[0]
        # print(s_next_pred, sum((s_next - s_next_pred)**2))
        return -sum((s_next - s_next_pred)**2)


def train(model, target_model, gamma, optimizer, batch_size, replay_time):
    s, a, r, s_next, done_flag = model.sample_batch(batch_size)
    with tf.GradientTape() as tape:
        q_value = model(s)
        a_index = tf.expand_dims(tf.range(batch_size), axis=1)
        a_index = tf.concat([a_index, a], axis=1)
        q_v = tf.gather_nd(q_value, a_index)
        q_v = tf.expand_dims(q_v, axis=1)

        q_target = target_model(s_next)
        q_target = r + gamma * tf.reduce_max(q_target,axis = 1,keepdims = True) * done_flag # for DQN


        a = tf.cast(a, dtype = tf.float32)
        b_loss =tf.reduce_sum((model.back_net(tf.concat([a, s], axis = 1)) - s_next) ** 2, axis=1)
        b_loss = tf.expand_dims(b_loss, axis=1)
        a_prob = model(s)
        f_loss = tf.nn.softmax_cross_entropy_with_logits(labels=a_prob, logits= model.for_net(tf.concat([s, s_next], axis = 1)))
        f_loss = tf.expand_dims(f_loss, axis=1)
        q_loss = (q_target - q_v) ** 2
        # print(b_loss.shape, f_loss.shape, q_loss.shape)
        loss = tf.reduce_mean((q_loss + b_loss + f_loss), keepdims=False)

        # loss_fn = losses.Huber()
        # loss  = loss_fn(q_target, q_v)
        # print(loss)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))



# Hyperparameter
MEM_LEN = 30000
LEARNING_RATE = 1e-3
EPSILON_INIT = 0.1
UPDATE_INTERVAL = 50
BATCH_SIZE = 32
input_size = 4
output_size = 2
MAX_EPOCH = 10000
GAMMA = 0.98



if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    model = DQN(input_size, output_size, MEM_LEN)
    target_model = DQN(input_size, output_size, MEM_LEN)
    for raw, target in zip(model.variables, target_model.variables):
        target.assign(raw)
    epsilon = EPSILON_INIT
    train_flag = False
    step_total = 0
    for i in range(MAX_EPOCH):
        epsilon = max(epsilon*0.999, 0.01)
        s = env.reset()
        done = False
        score = 0.
        while not done:
            step_total += 1
            # env.render()
            a = model.choose_action(s, epsilon)
            s_next, r, done, info = env.step(a)
            reward_icm = model.cal_ICMreward(s, s_next, a)
            # print(reward_icm)
            score += r
            r += reward_icm
            done_flag = 0. if done else 1.
            model.save_trans((s, a, r, s_next, done_flag))

            s = s_next

            if step_total > 1000:
                train_flag = True
                train(model, target_model, GAMMA, optimizers.Adam(LEARNING_RATE), BATCH_SIZE, 20)

            if done:
                print("Epoch:{}  score:{}  train_flag:{}  epsilon:{}".format(i, score, train_flag, epsilon))

        if i % UPDATE_INTERVAL == 0 and i > 0:
            for raw, target in zip(model.variables, target_model.variables):
                target.assign(raw)
            print("Target net cover!")




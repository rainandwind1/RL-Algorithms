import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers
import numpy as np
import gym


class PPO(keras.Model):
    def __init__(self, input_size, output_size):
        super(PPO,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mem = []
        self.net = keras.Sequential([
            layers.Dense(256, activation = tf.nn.relu),
            layers.Dense(128, activation = tf.nn.relu)
        ])
        self.policy = layers.Dense(self.output_size)
        self.critic = layers.Dense(1)

    def get_policy(self, inputs, dim):
        op = self.net(inputs)
        ap = self.policy(op)
        action_prob = tf.nn.softmax(ap, axis = dim)
        return  action_prob


    def get_critic(self, inputs):
        q_value = self.critic(inputs)
        return q_value


    def save_trans(self, transition):
        self.mem.append(transition)

    def package_trans(self):
        s_ls, a_ls, r_ls, s_next_ls, a_prob_ls, done_flag_ls = [], [], [], [], [], []
        for trans in self.mem:
            s, a, r, s_next, a_prob, done_flag = trans
            s_ls.append(s)
            a_ls.append([a])
            r_ls.append([r])
            s_next_ls.append(s_next)
            a_prob_ls.append([a_prob])
            done_flag_ls.append([done_flag])

        self.mem = []
        return tf.constant(s_ls, dtype = tf.float32),\
                tf.constant(a_ls, dtype = tf.int32),\
                tf.constant(r_ls, dtype = tf.float32),\
                tf.constant(s_next_ls, dtype = tf.float32),\
                tf.constant(a_prob_ls, dtype = tf.float32),\
                tf.constant(done_flag_ls,  dtype = tf.float32)


def train_net(model, optimizer, gamma, epsilon, lmd, k_epoch):
    s, a, r, s_next, a_prob, done_flag = model.package_trans()
    for epo_i in range(k_epoch):
        td_target =  r + gamma * model.get_critic(s_next) * done_flag
        td_error = td_target - model.get_critic(s)
        td_error = td_error.numpy()
        advantage_ls = []
        advantage = 0.
        for error in td_error[::-1]:
            advantage = gamma * lmd * advantage + error[0]
        advantage_ls.append(advantage)
        advantage_ls.reverse()
        with tf.GradientTape() as tape:
            advantage = tf.constant(advantage_ls, dtype = tf.float32)

            policy = model.get_policy(s, 1)
            index = tf.expand_dims(tf.range(a.shape[0]), 1)
            # print(index.shape, a.shape)
            a_index = tf.concat([index, a], axis = 1)
            policy = tf.gather_nd(policy, a_index)
            policy = tf.expand_dims(policy, 1)

            ratio = tf.exp(tf.math.log(policy) - tf.math.log(a_prob))
            surr1 = ratio * advantage_ls
            surr2 = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon) * advantage
            loss = -tf.minimum(surr1, surr2) + losses.MSE(model.get_critic(s), td_target)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


# Hyperparameters
LEARNING_RATE = 0.0003
GAMMA = 0.98
K_EPOCH = 3
LAMBDA = 0.95
EPS_CLIP = 0.1
T_HORIZON = 20
MAX_EPOCH = 10000


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    model = PPO(4, 2)
    optimizer = optimizers.Adam(lr=LEARNING_RATE)
    for epo_i in range(MAX_EPOCH):
        obs = env.reset()
        done = False
        score = 0.
        while not done:
            for step in range(T_HORIZON):
                a_prob = model.get_policy(tf.expand_dims(tf.constant(obs, dtype = tf.float32), axis = 0), 1)

                a = tf.random.categorical(a_prob, 1, dtype = tf.int32).numpy()[0, 0]
                obs_next, r, done, info = env.step(a)
                done_flag = 0. if done else 1.
                model.save_trans([obs, a, r, obs_next, a_prob.numpy()[0, a], done_flag])

                obs = obs_next
                score += r

                if done:
                    print("Epoch:{}  score:{}".format(epo_i, score))
                    break
            train_net(model, optimizer, GAMMA, EPS_CLIP, LAMBDA, K_EPOCH)

    env.close()







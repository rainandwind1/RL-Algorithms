import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizer,losses
import numpy as np
import collections
import gym
import copy
import threading
import multiprocessing

# A3C
env = gym.make('CartPole-v1')

class ActorCritic(keras.Model):
    def __init__(self,state_size,action_size):
        super(ActorCritic,self).__init__()
        self.state_size = state_size
        self.actor_size = action_size
        # 策略网络Actor
        self.dense1 = layers.Dense(128,activation = 'relu')
        self.policy_logits = layers.Dense(action_size)
        # V网络 Critic
        self.dense2 = layers.Dense(128,activation = 'relu')
        self.values = layers.Dense(1)


    def call(self,inputs):
        # 获得策略分布pi
        pi = self.dense1(inputs)
        pi_logits = self.policy_logits(pi)
        # 或得V(s)
        V = self.dense2(inputs)
        value = self.values(V)
        
        return pi_logits,value

class Worker(threading.Thread):
    global_episode = 0
    global_avg_return = 0
    def __init__(self,server,opt,result_queue,idx):
        super(Worker,self).__init__()
        self.result_queue = result_queue # 共享队列
        self.server = server # 中央模型
        self.opt = opt # 中央优化器
        self.client = ActorCritic(4,2) # 线性私有网络
        self.worker_idx = idx # 线程id
        self.env = gym.make('CartPole-v1').unwrapped
        self.ep_loss = 0.0

    def run(self):
        total_step = 1
        mem = Memory()
        while Worker.global_episode < 400:
            current_state = self.env.reset()
            mem.clear()
            ep_reward = 0.
            ep_steps = 0
            self.ep_loss = 0
            time_count = 0
            done = False
            while not done:
                # 获得Pi，未经过softmax
                logits, _ = self.client(tf.constant(current_state[None,:],dtype = tf.float32))
                probs = tf.nn.softmax(logits)
                action = np.random.choice(2,p=probs.numpy()[0])
                new_state, reward, done, info = self.env.step(action)
                if done:
                    reward = -1
                ep_reward += reward
                mem.store(current_state,action,reward)
                if time_count == 20 or done:
                    with tf.GradientTape() as tape:  # 梯度记录
                        total_loss = self.compute_loss(done,new_state,mem)
                    self.ep_loss += float(total_loss)
                    grads = tape.gradients(total_loss,self.client.trainable_weights)
                    self.opt.apply_gradients(zip(grads,self.client.trainable_weights))

                    self.clients.set_weights(self.server.get_weights())
                    mem.clear()
                    time_count = 0
                    if done:
                        Worker.global_avg_return = \ 
                        Worker.global_episode += 1
                    ep_steps += 1
                    time_count += 1
                    current_state = new_state
                    total_step += 1
                self.result_queue.put(None) # 结束线程











            
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses
import numpy as np
import collections
import random
import matplotlib.pyplot as plt


# Hyperparameters
# Replay_time = 30


# DDQN Double Deep Q-learning
class DDQN(keras.Model):
    # 动作值函数网络
    def __init__(self,output_size,memory_len):
        super(DDQN,self).__init__()
        self.output_size = output_size
        self.fc1 = layers.Dense(128,kernel_initializer = "random_uniform")
        self.fc2 = layers.Dense(256,kernel_initializer = "random_uniform")
        self.fc3 = layers.Dense(256,kernel_initializer = "random_uniform")
        self.fc4 = layers.Dense(output_size,kernel_initializer = "random_uniform")
        self.memory_len = memory_len
        self.memory_list = collections.deque(maxlen=memory_len)
    
    def call(self,inputs,training = None):
        fc1 = tf.nn.relu(self.fc1(inputs))
        fc2 = tf.nn.relu(self.fc2(fc1))
        fc3 = tf.nn.relu(self.fc3(fc2))
        output = self.fc4(fc3)
        return output
    
    def sample_action(self,state,epsilon):
        state = tf.constant(state,dtype = tf.float32)
        state = tf.expand_dims(state,axis = 0)
        action_value = self(state)[0]
        lucky_coin = np.random.rand()
        if lucky_coin > epsilon:
            return int(tf.argmax(action_value))
        else:
            return np.random.randint(0,self.output_size)
    
    # 经验回放部分
    # 存储记忆 (St,At,Rt,St+1,done)
    def save_memory(self,transition):
        self.memory_list.append(transition)
    
    # 采样
    def sample_memory(self,n):
        mini_batch = random.sample(self.memory_list,n)
        s_ls,a_ls,r_ls,s_next_ls,done_mask_ls = [],[],[],[],[]
        for trans in mini_batch:
            s,a,r,s_next,done_mask = trans
            s_ls.append(s)
            a_ls.append([a])
            r_ls.append([r])
            s_next_ls.append(s_next)
            done_mask_ls.append([done_mask])
        return tf.constant(s_ls,dtype = tf.float32),\
            tf.constant(a_ls,dtype = tf.int32),\
            tf.constant(r_ls,dtype = tf.float32),\
            tf.constant(s_next_ls,dtype = tf.float32),\
            tf.constant(done_mask_ls,dtype = tf.float32)

# 训练函数
def train(q_net,q_target,optimizer,batch_size,gamma,loss_list,Replay_time):
    huber = losses.Huber()
    for i in range(Replay_time):
        s,a,r,s_next,done_flag = q_net.sample_memory(batch_size)
        with tf.GradientTape() as tape:
            # Q_value
            qa_out = q_net(s)
            a_index = tf.expand_dims(tf.range(a.shape[0]),axis = 1)
            a_index = tf.concat([a_index,a],axis = 1)
            q_a = tf.gather_nd(qa_out,a_index)
            q_a = tf.expand_dims(q_a,axis = 1)
            
            # Q_target_value
            qtarget_out = q_target(s_next)
            
            qtarget_out = tf.reduce_max(qtarget_out,axis = 1,keepdims = True) # for DQN

            # a_target = tf.argmax(qa_out,axis = 1)
            # a_target = tf.reshape(tf.cast(a_target,dtype = tf.int32),shape = (batch_size,1))
            # a_target_index = tf.expand_dims(tf.range(a_target.shape[0]),axis = 1)
            # a_target_index = tf.concat([a_target_index,a_target],axis = 1)
            # qtarget_out = tf.gather_nd(qtarget_out,a_target_index)
            # qtarget_out = tf.expand_dims(qtarget_out,axis=1)                  # for DDQN

            q_t = r + gamma*qtarget_out*done_flag

            # loss
            loss = huber(q_a,q_t)
            loss_list.append(loss)
        grads = tape.gradient(loss,q_net.trainable_variables)
        optimizer.apply_gradients(zip(grads,q_net.trainable_variables))


# 绘制结果
def plot_curse(target_list,loss_list):
    figure1 = plt.figure()
    plt.grid()
    X = []
    for i in range(len(target_list)):
        X.append(i)
    plt.plot(X,target_list,'-r')
    plt.xlabel('epoch')
    plt.ylabel('score')

    figure2 = plt.figure()
    plt.grid()
    X = []
    for i in range(len(loss_list)):
        X.append(i)
    plt.plot(X,loss_list,'-b')
    plt.xlabel('train step')
    plt.ylabel('loss')
    plt.show()

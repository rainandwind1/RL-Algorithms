import torch
from torch import nn
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from torch.autograd import Variable
import random
# for i,j in zip([1,2,3],[4,5,6]):
#     print(i,j)
# a = [-1,1,1]
# print(abs(a))
# huber = nn.SmoothL1Loss() 
# a = torch.tensor([[1.],[2.],[3.]])
# b = torch.tensor([[3.],[4.],[5.]])
# c = huber(a,b)
# print(c)
# huber = losses.Huber()
# a = tf.constant([[1.],[2.],[3.]])
# b = tf.constant([[2.],[3.],[4.]])
# c = huber(a,b)
# print(c)


#
# a = torch.tensor([[1,2],[2,3],[3,4]])
# b = torch.tensor([[3.],[4.],[5.]])
# a = torch.max(a,dim=1,keepdim=True)[0]
# b = 5
# a = random.choice(range(5))
# action_p = [0.1,0.9]
# a = np.random.choice(range(2),1,p=action_p)
# loss_m = np.load("E:\Grade_two\作业\强化学习\作业二\FTG4.50_rlhomework\info" + "\Loss_AC_ep" + str(60) + ".npy")
# loss_list = loss_m.tolist()
# print(loss_list)


from multiprocessing import Process

def func(i):
    print("process{}".format(i))

if __name__ == "__main__":
    process_list = []
    for i in range(5):
        p = Process(target=func, args=(i, ))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()



# class Proc(Process):
#     def __init__(self, name):
#         super(Process, self).__init__()
#         self.name = name
#
#     def run(self):
#         print("This is process {}".format(self.name))
#
#
# if __name__ == "__main__":
#     p_res = []
#     for i in range(5):
#         p = Proc(str(i))
#         p.start()
#         p_res.append(p)
#
#     for p in p_res:
#         p.join()




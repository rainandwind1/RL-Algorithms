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



a = torch.tensor([[1,2],[2,3],[3,4]])
b = torch.tensor([[3.],[4.],[5.]])
a = torch.max(a,dim=1,keepdim=True)[0]
b = 5
a = random.choice(range(5))
action_p = [0.1,0.9]
a = np.random.choice(range(2),1,p=action_p)
print(int(a))


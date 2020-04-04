import torch
from torch import nn
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from torch.autograd import Variable
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
print(a)


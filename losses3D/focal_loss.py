from keras import backend as K
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# Compatible with tensorflow backend

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
	return focal_loss_fixed


# pre = torch.randn(16, 1, 64, 64, 32)
# size = (16, 64, 64, 32)
# target = torch.ones(size=size, dtype=torch.int64)
# # target=torch.one()
# # target=target.int64()
# loss = focal_loss(gamma=2., alpha=.25)
# loss_score = loss(pre, target)

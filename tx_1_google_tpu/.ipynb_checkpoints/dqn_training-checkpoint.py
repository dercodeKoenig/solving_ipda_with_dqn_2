from google_tpu_dqn import DQNAgent
from environment import environment, candle_class
from transformer_layer import TransformerBlock, PositionEmbedding

import time
import os
import tensorflow as tf
import pandas as pd    
import numpy as np
from collections import deque
import random
import cv2
import pickle

data_dir = "../archive"
name = "dqn_trading_transformer_small"
#resume = True
resume = False

warmup_parallel = 8
train_parallel = 8
warmup_steps = 1000

#for dqn
lr = 0.0001
memory_size = 20000
gamma = 0.95
exploration = 0.02
target_model_sync = 100
batch_size = 64

#for environment
dlen = 120
pos_size = 0.02 * 100000
comm = 15/100000
res_high = 100




tf.keras.backend.clear_session()
tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
strategy = tf.distribute.experimental.TPUStrategy(tpu)



def proc_chart(x):
    #x1 = image
    #x2 = time
    x1 = x[::, :-1, :]
    x2 = x[::,-1,:]

    x1 = tf.keras.layers.Reshape((res_high, dlen+1, 1))(x1)
    
    x5 = tf.keras.layers.Conv2D(64, 9,activation="relu", padding="same")(x1)
    x1 = tf.keras.layers.Concatenate()([x1,x5])
    x1 = tf.keras.layers.Dense(64)(x1)
    
    x1 = tf.transpose(x1,perm=[0, 2, 1, 3])
    x1 = tf.keras.layers.Reshape((dlen+1, res_high*x1.shape[-1]))(x1)
    x2 = tf.keras.layers.Reshape((dlen+1, 1))(x2)
    x1 = tf.keras.layers.Concatenate()([x1,x2])
    
    x1 = tf.keras.layers.Dense(512)(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Dense(512)(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Dense(128)(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.LayerNormalization()(x1)
    
    
    x1 = PositionEmbedding(dlen+1, x1.shape[-1])(x1)
    x1 = TransformerBlock(x1.shape[-1], 8, 256)(x1)
    x1 = TransformerBlock(x1.shape[-1], 8, 256)(x1)
    x1 = TransformerBlock(x1.shape[-1], 8, 256)(x1)
    x1 = TransformerBlock(x1.shape[-1], 8, 256)(x1)

    x1 = tf.keras.layers.Dense(512)(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Dense(512)(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    #x1 = tf.keras.layers.GlobalAveragePooling1D()(x1)
    x1 = tf.keras.layers.GRU(512)(x1)
    
    x1 = tf.keras.layers.Dense(1024,activity_regularizer=tf.keras.regularizers.L2(0.00001))(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Dense(1024,activity_regularizer=tf.keras.regularizers.L2(0.00001))(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Dense(1024,activity_regularizer=tf.keras.regularizers.L2(0.00001))(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Dense(256,activity_regularizer=tf.keras.regularizers.L2(0.00001))(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    #x1 = tf.keras.layers.LayerNormalization()(x1)
    return x1
    
with strategy.scope():
    input_m15 = tf.keras.layers.Input(shape = (res_high+1, dlen+1))
    input_h1 = tf.keras.layers.Input(shape = (res_high+1, dlen+1))
    input_h4 = tf.keras.layers.Input(shape = (res_high+1, dlen+1))
    input_d1 = tf.keras.layers.Input(shape = (res_high+1, dlen+1))
    
    x1 = proc_chart(input_m15)
    x2 = proc_chart(input_h1)
    x3 = proc_chart(input_h4)
    x4 = proc_chart(input_d1)
    
    input_net_position = tf.keras.layers.Input(shape = (1))


    x = tf.keras.layers.Concatenate()([x1,x2,x3,x4,input_net_position])
    
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    outputs = tf.keras.layers.Dense(2, activation = "linear", use_bias=False, dtype="float32")(x)
    model = tf.keras.Model([input_m15,input_h1,input_h4, input_d1, input_net_position], outputs)
    
model.summary()




with strategy.scope():
    opt = tf.keras.optimizers.Adam(lr)



agent = DQNAgent(
    model = model, 
    strategy = strategy,
    n_actions = 2, 
    memory_size = memory_size, 
    gamma=gamma,
    optimizer = opt,
    batch_size = batch_size, 
    target_model_sync = target_model_sync,
    exploration = exploration,
    name=name+".h5")

if resume:
	print("loading weights...")
	agent.load_weights()
    
    
    
x = [environment(data_dir, dlen, res_high, comm, pos_size) for _ in range(warmup_parallel)]
print("warmup...")
n = warmup_steps
agent.train(num_steps = n, envs = x, warmup = n, log_interval = n)
len(agent.memory)



x = [environment(data_dir, dlen, res_high, comm, pos_size) for _ in range(train_parallel)]
print("training...")
n = 1000000000
agent.train(num_steps = n, envs = x, warmup = 0, log_interval = 1000)
print("done")
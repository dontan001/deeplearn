#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# 引入集合模块，应用于词频统计
import collections
import tensorflow as tf
import random
import numpy as np


# 1.读取数据
content = ''
with open("data/belling_the_cat.txt") as f:
    content = f.read()
    # print(content)

# 2. 根据每个符号（单词+标点符号）出现频率为其分配一个对应的整数
# (1) 文本数据根据空格做切割，转换为每一个符号
words = content.split()

# (2) 构建字典
def build_dataset(words):
    count = collections.Counter(words).most_common()
    # print(count)
    dictionary = dict()
    for word,_ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    return dictionary, reverse_dictionary

# 3. 构建RNN网络
dictionary, reverse_dictionary = build_dataset(words)
vocab_size = len(dictionary)
n_input = 3  # 输入数据个数
n_hidden = 512  # 神经元个数
batch_size = 20  # 批次
# weight = tf.get_variable('weight_out', [n_hidden, vocab_size], initializer=tf.truncated_normal([n_hidden, vocab_size], stddev=0.1))
# bias = tf.get_variable('bias_out', [vocab_size])
weight = tf.get_variable('weight_out', [n_hidden, vocab_size], initializer=tf.random_normal_initializer)
bias = tf.get_variable('bias_out', [vocab_size], initializer=tf.random_normal_initializer)

def RNN(x,weight,bias):
    # reshape to [1, n_input]
    # x [n_input] like['long','long','ago'] -> [number,number,number]
    x = tf.reshape(x,[-1,n_input])
    x = tf.split(x,n_input,1)

    # 构建单层lstm
    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    # 运行网络
    # 输出结果，细胞状态
    outputs, states = tf.nn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # 执行softmax操作（wx+b）
    return tf.matmul(outputs[-1],weight) + bias

# 数据转换
def build_data(offset):
    if offset + 3 > vocab_size:
        offset = random.randint(0, vocab_size - n_input)
    symbols_in_key = [[dictionary[str(words[i])]] for i in range(offset,offset+n_input)]
    symbols_out_onehot = np.zeros([vocab_size], dtype = np.float32)
    symbols_out_onehot[dictionary[str(words[offset + n_input])]] = 1.0
    return symbols_in_key, symbols_out_onehot

# 构建损失函数
x = tf.placeholder(tf.float32,[None, n_input,1])
y = tf.placeholder(tf.float32,[None, vocab_size])
pred = RNN(x,weight,bias)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# 优化函数
optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)

# 训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        x_train,y_train = [],[]
        for b in range(batch_size):
            new_x, new_y = build_data(random.randint(0, vocab_size))
            x_train.append(new_x)
            y_train.append(new_y)
        # print(x_train)
        _opt = sess.run(optimizer, feed_dict={x: np.array(x_train), y: np.array(y_train)})
        if i % 100 == 0:
            print(sess.run(cost, feed_dict={x: np.array(x_train), y: np.array(y_train)}))

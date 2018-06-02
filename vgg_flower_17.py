#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from tflearn.datasets import oxflower17
import tensorflow as tf
# mnist oxflower17 cifar10 cifar100
X,Y = oxflower17.load_data(one_hot = True)
print(X.shape) # 224,224,3
print(Y.shape) # 17

# 图像分类
# 0.读取数据
# 1.设计超参数
# （1）训练有关的参数
train_epoch = 1000 # 训练迭代次数（训练数据遍历完一次，表示一次迭代）
batch_size = 10 # 每个批次训练多少数据
learning_rate = tf.placeholder(tf.float32) # 准备设计一个动态的学习率
display_epoch = 100 # 每迭代100次，进行一次评估
n_class = Y.shape[1]

# （2）网络有关的参数
# 权重
weight = {
    'wc1_1': tf.Variable(tf.random_normal([3,3,3,64])),
    'wc2_1': tf.Variable(tf.random_normal([3,3,64,128])),
    'wc3_1': tf.Variable(tf.random_normal([3,3,128,256])),
    'wc3_2': tf.Variable(tf.random_normal([3,3,256,256])),
    'wc4_1': tf.Variable(tf.random_normal([3,3,256,512])),
    'wc4_2': tf.Variable(tf.random_normal([3,3,512,512])),
    'wc5_1': tf.Variable(tf.random_normal([3,3,512,512])),
    'wc5_2': tf.Variable(tf.random_normal([3,3,512,512])),
    'wfc_1': tf.Variable(tf.random_normal([7*7*512,4096])),
    'wfc_2': tf.Variable(tf.random_normal([4096,4096])),
    'wfc_3': tf.Variable(tf.random_normal([4096,n_class])),
}
# 偏置量
biase = {
    'bc1_1': tf.Variable(tf.random_normal([64])),
    'bc2_1': tf.Variable(tf.random_normal([128])),
    'bc3_1': tf.Variable(tf.random_normal([256])),
    'bc3_2': tf.Variable(tf.random_normal([256])),
    'bc4_1': tf.Variable(tf.random_normal([512])),
    'bc4_2': tf.Variable(tf.random_normal([512])),
    'bc5_1': tf.Variable(tf.random_normal([512])),
    'bc5_2': tf.Variable(tf.random_normal([512])),
    'bfc_1': tf.Variable(tf.random_normal([4096])),
    'bfc_2': tf.Variable(tf.random_normal([4096])),
    'bfc_3': tf.Variable(tf.random_normal([n_class])),
}
x = tf.placeholder(tf.float32, [None,224,224,3])  # 输入的数据
y = tf.placeholder(tf.float32, [None,n_class])  # 标签数据

# 2.设计网络
def vgg_network():
    # input reshape  输入层的预处理
    # conv_1
    # filter 卷积核
    # strides 步长
    net = tf.nn.conv2d(input=x, filter=weight['wc1_1'], strides=[1,1,1,1], padding='VALID') #卷积
    print("Conv", net)
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc1_1'])) #激励
    print("Relu", net)
    net = tf.nn.lrn(net) # 局部归一化
    print("LRN", net)
    # ksize 窗口大小
    # strides 步长
    net= tf.nn.max_pool(value=net, ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID') # 池化
    print("Max pooling", net)

    # conv_2
    net = tf.nn.conv2d(net, weight['wc2_1'], [1, 1, 1, 1], padding='VALID')
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc2_1']))
    net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1])

    # conv_3
    net = tf.nn.conv2d(net, weight['wc3_1'], [1, 1, 1, 1], padding='VALID')
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc3_1']))
    net = tf.nn.conv2d(net, weight['wc3_2'], [1, 1, 1, 1], padding='VALID')
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc3_2']))
    net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

    # conv_4
    net = tf.nn.conv2d(net, weight['wc4_1'], [1, 1, 1, 1], padding='VALID')
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc4_1']))
    net = tf.nn.conv2d(net, weight['wc4_2'], [1, 1, 1, 1], padding='VALID')
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc4_2']))
    net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

    # conv_5
    net = tf.nn.conv2d(net, weight['wc5_1'], [1, 1, 1, 1], padding='VALID')
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc5_1']))
    net = tf.nn.conv2d(net, weight['wc5_2'], [1, 1, 1, 1], padding='VALID')
    net = tf.nn.leaky_relu(tf.nn.bias_add(net, biase['bc5_2']))
    net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

    # 拉伸
    net = tf.reshape(net,shape=[-1, weight['wfc_1'].get_shape().to_list()[0]])

    # 全连接（BP网络）
    # fc1
    net = tf.nn.relu(tf.matmul(net, weight['wfc_1']) + biase['bfc_1'])
    # fc2
    net = tf.nn.relu(tf.matmul(net, weight['wfc_2']) + biase['bfc_2'])
    # out
    # return tf.nn.softmax(tf.matmul(net, weight['wfc_3']) + biase['bfc_3'])
    return tf.matmul(net, weight['wfc_3']) + biase['bfc_3']


# 3.设计损失函数和优化器，建立评估函数
pred = vgg_network()
loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y) #损失函数
# 用于sigmoid
# tf.nn.sigmoid_cross_entropy_with_logits()
# 优化器由分类器和损失函数决定
opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss) #优化器
# one_hot 17位
# [True,False,True,...]  X
# True
acc_tf = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
acc = tf.reduce_mean(tf.cast(acc_tf,tf.float32))

# 4.训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 初始化全局变量
    base_lr = 0.01

    for epoch in range(train_epoch):
        # batch_size=10,total_data=1360 => total_batch
        # 取数据
        total_batch = X.shape[0]//batch_size
        for i in range(total_batch):
            X_train,Y_train = X[i*batch_size:i*batch_size+batch_size],Y[i*batch_size:i*batch_size+batch_size]
            # learning_rate = batch_size*((1-epoch)/train_epoch)**2
            sess.run(opt,feed_dict={x:X_train,y:Y_train,learning_rate:base_lr})
            if (i+1)*(epoch+1)%display_epoch ==0:
                # 进行评估
                cost, accuaray = sess.run(opt,feed_dict={x:X_train,y:Y_train,learning_rate:base_lr})
                print("Step: %d, loss: %d, acc: %d" % (str(epoch)+"-"+str(i),cost,accuaray))

                # 动态修改学习率
                learning_rate = base_lr * ((1 - epoch) / train_epoch) ** 2

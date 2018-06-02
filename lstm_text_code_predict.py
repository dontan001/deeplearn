#引入集合模块，应用于词频统计
import collections
import tensorflow as tf
import random
import numpy as np
# 1.读取数据
content=''
with open('belling_the_cat.txt') as f:
    content=f.read()

# print(content)
# 2.根据每个符号（单词+标点符号）出现频率为其分配一个对应的整数
# （1）文本数据根据空格做切割，转换为每一个符号
words=content.split()

# （2）构建字典
def bulid_dataset(words):
    count=collections.Counter(words).most_common()
    dictionary=dict()
    for word,_ in count:
        dictionary[word]=len(dictionary)
    reverse_dictionary=dict(zip(dictionary.values(),dictionary.keys()))
    return dictionary,reverse_dictionary
dictionary,reverse_dictionary=bulid_dataset(words)
#3.构建RNN网络
vocab_size=len(dictionary)
n_input=3 #输入数据个数
n_hidden=512 # 神经元个数

weight=tf.get_variable('weight_out',[2*n_hidden,vocab_size],initializer=tf.random_normal_initializer)
bias=tf.get_variable('bias_out',[vocab_size],initializer=tf.random_normal_initializer)

def RNN(x,weight,bias):
    # reshape to [-1,n_input]
    # x [n_input] [[2,3,4],[3,4,5],[6,7,8]]
    x=tf.reshape(x,[-1,n_input])
    # [[2][3][4]]
    x=tf.split(x,n_input,1)
    # 构建单层双向lstm
    rnn_cell_format=tf.nn.rnn_cell.BasicLSTMCell(n_hidden,state_is_tuple=True,forget_bias=1.0)
    run_cell_backmat=tf.nn.rnn_cell.BasicLSTMCell(n_hidden,state_is_tuple=True,forget_bias=1.0)
    # 运行网络
    # 输出结果，细胞状态
    # 生成双向rnn
    outputs, output_state_fw, output_state_bw=tf.nn.static_bidirectional_rnn(rnn_cell_format,run_cell_backmat,x,dtype=tf.float32)

    # 执行wx+b
    return tf.matmul(outputs[-1],weight)+bias

x=tf.placeholder(tf.float32,[None,n_input,1])
pred=RNN(x,weight,bias)

# 预测
with tf.Session() as sess:
    # 模型载入
    saver=tf.train.Saver()
    saver.restore(sess,saver.recover_last_checkpoints('./models'))
    while True:
        start_sentence=input("Please input %s words:"%n_input)
        words=start_sentence.strip().split()
        if len(words)==n_input:
            continue
        symbols_in_keys=[dictionary[str(words[i])] for i in range(len(words))]
        for _ in range(32):# 生成32个单词
            keys=np.reshape(np.array(symbols_in_keys),[-1,n_input,1])
            onehot_pred=sess.run(pred,feed_dict={x:keys})
            #得到对应的词向量最大值的下标
            onehot_pred_index=int(np.argmax(onehot_pred[0],1))
            # 使用反向字典真实的单词
            start_sentence='%s %s'%(start_sentence,reverse_dictionary[onehot_pred_index])
            symbols_in_keys=symbols_in_keys[1:]
            symbols_in_keys.append(onehot_pred_index)
        print(start_sentence)
        break
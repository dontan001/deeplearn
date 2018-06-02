#引入集合模块，应用于词频统计
import collections
import tensorflow as tf
import random
import numpy as np
# 1.读取数据
content=''
with open('data/belling_the_cat.txt') as f:
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
batch_size=20
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

# 数据转换
def build_data(offset):
    while offset+n_input>vocab_size:
        offset=random.randint(0,vocab_size-n_input)
    symbols_in_key=[[dictionary[str(words[i])]] for i in range(offset,offset+n_input)]
    symols_out_onehot=np.zeros([vocab_size],dtype=float)
    symols_out_onehot[dictionary[str(words[offset+n_input])]]=1.0
    return symbols_in_key,symols_out_onehot

# 构建损失函数
x=tf.placeholder(tf.float32,[None,n_input,1])
y=tf.placeholder(tf.float32,[None,vocab_size])
pred=RNN(x,weight,bias)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=y))
# 优化函数
optimizer=tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)
#准确率
correct_preb=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuary=tf.reduce_mean(tf.cast(correct_preb,tf.float32))

#训练
with tf.Session() as sess:
    saver=tf.train.Saver(tf.global_variables())
    sess.run(tf.global_variables_initializer())
    for i in range(50000):
        x_train,y_train=[],[]
        for b in range(batch_size):
            new_x, new_y = build_data(random.randint(0, vocab_size))
            x_train.append(new_x)
            y_train.append(new_y)
        #print(x_train)
        _opt=sess.run(optimizer,feed_dict={x:np.array(x_train),y:np.array(y_train)})
        if i%100==0:
            # 显示当前的x的数据和真实y的值以及预测的值
            acc,out_pred=sess.run([accuary,pred], feed_dict={x: np.array(x_train), y: np.array(y_train)})
            symbols_in=[reverse_dictionary[word_index[0]] for word_index in x_train[0]]
            symbols_out=reverse_dictionary[int(np.argmax(y_train,1)[0])]
            pred_out = reverse_dictionary[int(np.argmax(out_pred, 1)[0])]
            print('Acc:%f'%acc)
            print('%s -[%s] vs [%s]'%(symbols_in,symbols_out,pred_out))
            if acc>0.95:
                saver.save(sess,'./model/lstm',global_step=i)


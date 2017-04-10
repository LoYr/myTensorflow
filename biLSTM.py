import tensorflow as tf
import numpy as np
import lyr.word2vec as word2vec

#先训练词向量
final_embeddings =  word2vec.run()


word_index = []

#找出词语对应的排名
def find_index(onlywhat):
    for word in onlywhat:
        index = word2vec.reverse_dictionary1.get(word)
        if index not in word_index:
            word_index.append(index)
    return word_index
find_index(word2vec.onlyword)
print("word_index:(按病历每个分好词语的排名): ",word_index)


#tab为分好的标签
print("tab",word2vec.tab)

labels = []
#将标签9类化
def y_labels(input):
    for label in input:
        if label == "O":
            labels.append([1, 0, 0, 0, 0, 0, 0, 0, 0])
        elif label == "B-DI":
            labels.append([0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif label == "I-DI":
            labels.append([0, 0, 1, 0, 0, 0, 0, 0, 0])
        elif label == "B-TR":
            labels.append([0, 0, 0, 1, 0, 0, 0, 0, 0])
        elif label == "I-TR":
            labels.append([0, 0, 0, 0, 1, 0, 0, 0, 0])
        elif label == "B-SY":
            labels.append([0, 0, 0, 0, 0, 1, 0, 0, 0])
        elif label == "B-TE":
            labels.append([0, 0, 0, 0, 0, 0, 1, 0, 0])
        elif label == "I-SY":
            labels.append([0, 0, 0, 0, 0, 0, 0, 1, 0])
        elif label == "I-TE":
            labels.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
    return labels

#y_labels为将所有标签9类化了
labels = y_labels(word2vec.tab)


#获取词向量
def getWordEmbedding():
    word_embedding = tf.nn.embedding_lookup(final_embeddings, word_index)
    return word_embedding
word_embedding = getWordEmbedding()


#获取字与词向量的index
def getAllWord(onlywhat):
    alist = [ [] for i in range(len(onlywhat))]
    zilist = []
    for i in range(len(onlywhat)):
        Index = word2vec.reverse_dictionary1.get(onlywhat[i])
        zilist.append(Index)
        for zi in onlywhat[i]:
            index = word2vec.reverse_dictionary1.get(zi)
            zilist.append(index)
        alist[i] = zilist
        zilist = []

    return alist
#(为一个列表，列表每一项第一个是词向量，后面为该词的每个字向量)
print("zilist: ", getAllWord(word2vec.onlyword)[:2])

#字词联合训练得到词向量
def findAllEmbedding(allword):
    final_embedding = [ [] for i in range(len(allword))]
    for j in range(len(allword)):
        embedding = tf.nn.embedding_lookup(final_embeddings, allword[j])
        #tf.reduce_mean(x,0) 以列来求和再平均， 1以行来求和平均
        all_embedding = tf.reduce_mean(embedding, 0)
        # embedding = tf.reshape(embedding,[-1])              #shape:[len(allword) * 128] 如 6*128 = 768
        # number = 0
        # for i in range(len(embedding)):
        #     number += tf.matmul(number, embedding[i])
        # all_embedding = [ x / (len(embedding) * 128) for x in number]

        final_embedding[j] = all_embedding
        all_embedding = []
    return final_embedding

#final_embedding: 字词联合的词向量










learning_rate = 1.0
batch_size = 138
display_step = 1000
epoch_step = 200000

n_input = 128
n_step = 9
n_class = 9
n_hidden = 256

x = tf.placeholder("float", [None, n_step, n_input])
y = tf.placeholder("float", [None, n_class])

w = tf.Variable(tf.random_normal([2 * n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))



# def mini_batch(all_data,batch_size):
#     data = all_data.eval()
#     mini_epoch = len(data) / (n_input * n_step * batch_size)
#     batch1 = data[:mini_epoch]
#     batch2 = data[mini_epoch + 1, 2 * mini_epoch]
#     batch3 = data[2 * mini_epoch:]
#     print("batch1:" + len(batch1) + ",batch2: " + len(batch2) + ",batch3: " + len(batch3))
#     return batch1,batch2,batch3



def BiLSTM(x_input, weights, biases):
    x_input = tf.transpose(x_input, [1, 0 ,2])
    x_input = tf.reshape(x_input, [-1, n_input])
    x_input = tf.split(x_input, n_input)

    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)

    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x_input, dtype = tf.float32)

    return tf.matmul(outputs[-1], weights) + biases

pred = BiLSTM(x, w, b)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    data = word_embedding.eval()           # data:（3726,128）
    mini_epoch = len(data) / (n_step * batch_size)
    batch1 = data[:mini_epoch]
    batch2 = data[mini_epoch + 1:2 * mini_epoch]
    batch3 = data[2 * mini_epoch:]

    for epoch in range(epoch_step):
        for i in range(int(mini_epoch)):
            if i == 0:
                batch_x = data[:1242]       # n_step * batch_size = 1242
            elif i == 1:
                batch_x = data[1242:1284]
            elif i == 2:
                batch_x = data[1284:]


            batch_x = tf.reshape(batch_x, [batch_size, n_step, n_input])
            batch_x = batch_x.eval()
            batch_y = labels

            sess.run(optimizer, feed_dict = {x:batch_x, y:batch_y})
            if epoch % display_step == 0:
                acc = sess.run(accuracy, feed_dict = {x: batch_x, y:batch_y})
                loss = sess.run(cost, feed_dict = {x:batch_x, y:batch_y})

                print("epoch: "+ epoch + ", acc: " + acc + ", loss: " + loss)

    print("finish")











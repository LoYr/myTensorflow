import math
import random
import collections
import numpy as np
import tensorflow as tf

with open('/tensorflow/lyr/train.txt','r',encoding='UTF-8') as f:
    lines = [line for line in f]
    Line = []
    chinese = []
    onlyword = []
    onlyzi = []
    tab = []
    for line in lines:
        line = line.strip().split()
        for i in range(len(line)):
            Line.append(line[i])

    for j in range(len(Line)):
        chinese.append(Line[j].split('/'))
        onlyword.append(chinese[j][0])
        tab.append(chinese[j][1])

    for word in onlyword:
        for i in range(len(word)):
            onlyzi.append(word[i])

    #加了词语进去
    onlyzi.extend(onlyword)

    print("ontlyword: ",onlyword)
    print("病历数据: ",chinese)
#print("原始数据：", Line)
print("data size: ",len(chinese))

vocabulary = 5000
def build_data(words):
    count = []
    count.extend(collections.Counter(words).most_common())
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    for word in words:
        index = dictionary[word]

        data.append(index)

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    reverse_dictionary1 = dict(zip(dictionary.keys(),dictionary.values()))
    return data, count, dictionary, reverse_dictionary,reverse_dictionary1

data, count, dictionary, reverse_dictionary,reverse_dictionary1 = build_data(onlyzi)
print("len:",len(dictionary))
print("data（以频数排序的编号）:",data)
print("count（频数统计）:",count)
print("dictionary:",dictionary)
print("reverse_dictionary（排名:单词）:",reverse_dictionary)
print("reverse_dictionary1（单词：排名）:",reverse_dictionary1)
print("most common words: ", count[:5])
print('sample data: ', data[:10],[reverse_dictionary[i] for i in data[:10]])

data_index = 0

#num_skips: 每个单词生成样本数   skip_window:单词最远距离联系的距离
#生成样本
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape = (batch_size), dtype = np.int32)          #将batch初始化为数组
    labels = np.ndarray(shape = (batch_size, 1), dtype = np.int32)      #将labels初始化为数组
    span = 2 * skip_window + 1                                          #span:一个窗口
    buffer = collections.deque(maxlen=span)                             #collections.deque：生成双向队列

    #生成滑动窗口
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    for i in range(batch_size // num_skips):
        target = skip_window
        target_to_avoid = [ skip_window ]
        for j in range(num_skips):
            while target in target_to_avoid:
                 target = random.randint(0, span -1)
            target_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    #print("batche: \n",batch,"labels: ",labels)
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

batch_size = 128
embedding_size = 128
learning_rate = 1.0
skip_window = 1
num_skips = 2

valid_size = 16
valid_window = 100
#随机从100个里面选16个
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
#训练时噪声单词数量
num_sampled = 2

train_inputs = tf.placeholder(tf.int32, [batch_size])
train_labels = tf.placeholder(tf.int32, [batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype = tf.int32)

embeddings = tf.Variable(tf.random_uniform([vocabulary, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

nce_weights = tf.Variable(tf.truncated_normal([vocabulary, embedding_size], stddev = 1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary]))

loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights,
                                     biases = nce_biases,
                                     labels = train_labels,
                                     inputs = embed,
                                     num_sampled = num_sampled,
                                     num_classes = vocabulary))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

# reduce_sum(x,1,keep_dims = true)  x = [[1,1,1],[1,1,1]]
# tf.reduce_sum(x,0):[2,2,2] 中间值为0时，竖着加 为1时，横着加
# tf.reduce_sum(x,1): [3,3]
# tf.reduce_sum(x,1,keep_dims = true) : [[3],[3]]
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1, keep_dims = True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b = True)

saver = tf.train.Saver()
init = tf.global_variables_initializer()

num_steps = 2001

def run():
    with tf.Session() as sess:
        sess.run(init)

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
            _, loss_val = sess.run([optimizer, loss],
                                   feed_dict={train_inputs: batch_inputs, train_labels: batch_labels})


            average_loss += loss_val

            if step % 2000 == 0:

                if step > 0:
                    average_loss /= 2000

                print("Average loss at step: ", step, ": ", average_loss)
                average_loss = 0

            #打印最相近的词
            # if step % 80000 == 0 and step > 0:
            #     sim = similarity.eval()
            #     for i in range(valid_size):
            #         valid_word = reverse_dictionary[valid_examples[i]]
            #         top_k = 8
            #         nearest = (-sim[i, :]).argsort()[1: top_k + 1]
            #         log_str = "Nearest to %s" % valid_word
            #         for k in range(top_k):
            #             close_word = reverse_dictionary[nearest[k]]
            #             log_str = "%s %s, " % (log_str, close_word)
            #
            #         print(log_str)

            #if step % 300000 == 0 and step > 0:
                #saver.save(sess, "D:/tensorflow/lyr/wordVector_model_save/model")

        # 最终词向量
        final_embeddings = normalized_embeddings.eval()

        #验证final_embedding 的顺序是否正确
        validEmbedding = [1,362,59,225]

        #print(tf.nn.embedding_lookup)只能打出tensor的shape,要打印具体 sess.run(tf.nn.embedding_lookup)
        valid_Embedding = sess.run(tf.nn.embedding_lookup(final_embeddings, validEmbedding))
        print("final embeddings ",final_embeddings)
        print("validEmbedding: ",valid_Embedding)
    return final_embeddings














import tensorflow as tf


learning_rate = 0.01
iters = 20000
batch_size = 64
display_step = 20

n_input = 361       # alphaGo data input (img shape: 19*19)
n_class = 361       # AlphaGo total classes (19x19=361 digits)
dropout = 0.681     # Dropout, probability to keep units 这里是随机概率当掉一些节点来训练，随你填，我一般用黄金分割点

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_class])
keep_prob = tf.placeholder("float32")

w = {
    "conv1": tf.Variable(tf.random_normal([3, 3, 1, 64])),
    "conv2": tf.Variable(tf.random_normal([3, 3, 64, 128])),
    "conv3": tf.Variable(tf.random_normal([3, 3, 128, 256])),
    "d1": tf.Variable(tf.random_normal([4 * 4 * 256, 1024])),
    "d2": tf.Variable(tf.random_normal([1024, 1024])),
    "out": tf.Variable(tf.random_normal([1024, n_class]))
}

b = {
    "conv1": tf.Variable(tf.random_normal([64])),
    "conv2": tf.Variable(tf.random_normal([128])),
    "conv3": tf.Variable(tf.random_normal([256])),
    "d1": tf.Variable(tf.random_normal([1024])),
    "d2": tf.Variable(tf.random_normal([1024])),
    "out": tf.Variable(tf.random_normal([n_class]))
}



def conv2d(x,w):
    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = "SAME")

def max_pool(l_input, k):
    return tf.max_pool(l_input, ksize = [1, k, k, 1], strides = [1, k, k, 1], padding = "SAME")

#归一化操作
def norm(l_input, lsize = 4):
    return tf.nn.lrn(l_input, lsize, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)


# 3个卷积层，2个全连接层
def alphgo(_x, _weights, _biases, _dropout):
    _x = _x.reshape([-1, 19, 19, 1])

    # convolution layer
    conv1 = tf.relu(conv2d(_x, _weights["conv1"]) + _biases["conv1"])
    pool1 = max_pool(conv1, k = 2)
    norm1 = norm(pool1, lsize = 4)
    norm1 = tf.nn.dropout(norm1, _dropout)

    # conv1 image show
    tf.image_summary(conv1)

    conv2 = tf.relu(conv2d(norm1, _weights["conv2"]) + _biases["conv2"])
    pool2 = max_pool(conv2, k = 2)
    norm2 = norm(pool2, lsize = 4)
    norm2 = tf.nn.dropout(norm2, _dropout)

    conv3 = tf.relu(conv2d(norm2, _weights["conv3"]) + _biases["conv3"])
    pool3 = max_pool(conv3, k = 2)
    norm3 = norm(pool3, lsize = 4)
    norm3 = tf.nn.dropout(norm3, _dropout)

    # fully connect layer
    dense1 = tf.reshape(norm3, [-1, 4 * 4 *1024])
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights["d1"]) + _biases["d1"])
    dense2 = tf.nn.relu(tf.matmul(dense1, _weights["d2"]) + _biases["d2"])

    out = tf.matmul(dense2, _weights["out"]) + _biases["out"]
    return out

pred = alphgo(x, w, b, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initialize()

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.train.SummaryWriter('/tmp/logs', graph_def=sess.graph_def)
    step = 1

    while step * batch_size < iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
            print("cot:", cost, "loss:", loss)
        step += 1

    print("finish")

    print("accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.image[:256], y: mnist.test.labels[:256], keep_prob: 1}))









import tensorflow as tf

learning_rate = 0.01
training_epoch = 30
batch_size = 100
display_step = 1

n_input = 784
n_class = 10

x = tf.placeholder("float", [None,n_input])
y = tf.placeholder("float",[None,n_class])

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = "SAME")         # strides = 步长

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")    #ksize = 滑动窗口大小  即 2 * 2 方格中选最大的一个

def multilayer_perceptron(x, weights, biases):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, weights["conv1"]) + biases["conv1"])
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, weights["conv2"]) + biases["conv2"])
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshapge(h_pool2, [-1, 7 * 7 * 64])      # 28 * 28 to 14 * 14 to 7 * 7 because 2 * 2 过滤了两次
    h_fcl = tf.nn.relu(tf.matmul(h_pool2_flat, weights["fcl"]) + biases["fcl_b"])

    out_layer = tf.matmul(h_fcl, weights["out"]) + biases["out_b"]

    return out_layer

weights = {
    "conv1": tf.Variable(tf.random_normal([5, 5, 1, 32])),      # 5,5:卷积核  1：卷基层上一层图的张数  32:卷积核个数 = 下一层图的张数
    "conv2": tf.Variable(tf.random_normal([5, 5, 32, 64])),
    "fcl": tf.Variable(tf.random_normal([7 * 7 * 64, 256])),
    "out": tf.Variable(tf.random_normal([256, n_class]))
}

biases = {
    "conv1": tf.Variable(tf.random_normal([32])),
    "conv2": tf.Variable(tf.random_normal([64])),
    "fcl_b": tf.Variable(tf.random_normal([256])),
    "out_b": tf.Variable(tf.random_noraml([n_class]))
}

pred = multilayer_perceptron(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epoch):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            _, cost = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            avg_cost += cost / total_batch

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("finish")

    correct_predict = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))
    print("accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))





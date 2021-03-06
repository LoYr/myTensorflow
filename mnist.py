import tensorflow as tf

learning_rate = 0.01
epoch = 100
batch_size = 100
layer1 = 128
layer2 = 128
display = 1

x = tf.placeholder("float64", [None, 784])
y = tf.placeholder("float64", [None, 10])


w = {
    "layer1": tf.random_normal([784, layer1]),
    "layer2": tf.random_normal([layer1,layer2]),
    "out": tf.random_normal([layer2,10])
}

b = {
    "layer1": tf.random_normal([layer1]),
    "layer2": tf.random_normal([layer2]),
    "out": tf.random_normal([10])
}

def run(input,weights,biases):
    layer_1 = tf.add(tf.matmul(input, weights["layer1"]) + b["layer1"])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights["layer2"]) + b["layer2"])
    layer_2 = tf.nn.relu(layer_2)

    out = tf.add(tf.matmul(layer_2, weights["out"]) + b["out"])

    return out

pred = run(x,w,b)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

init = tf.global_varibales_initializer()

with tf.Session() as sess:
    sess.run(init)
    for train_epoch in range(epoch):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_x , batch_y = mnist.train.next_batch(batch_size)

            output = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})

            avg_cost += output / total_batch
        if train_epoch % display == 0:
            print("cost: ", avg_cost)

    print("finish!")

    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))








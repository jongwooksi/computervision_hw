import os
import random
import numpy as np
import tensorflow as tf
import _pickle as cPickle
import matplotlib.pyplot as plt

DATA_PATH = './cifar-10-python'

random.seed(123)
tf.set_random_seed(123)
np.random.seed(123)

def unpickle(file):
    with open(os.path.join(DATA_PATH, file), 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin-1')
    return dict


def one_hot(labels, vals=10):
    n = len(labels)
    out = np.zeros((n, vals))
    out[range(n), labels ] = 1
    return out


def display_cifar(images, size):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([images[np.random.choice(n)] for i in range(size)])

    plt.imshow(im)
    plt.show()

def save_epoch(sess, current_epoch):
    epoch_ = tf.Variable(0, shape=(), dtype=tf.int16, name = "Epoch")
    current_epoch = sess.run(tf.compat.v1.assign(epoch_, current_epoch))


class CifarLoader(object):
    def __init__(self, source_files):
        self._source = source_files
        #self._i = 0
        self.images = None
        self.labels = None

    def load(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack([d['data'] for d in data])
        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1 ).astype(float)/255
        self.labels = one_hot(np.hstack([d['labels']for d in data]), 10)
        return self

    def next_batch(self, batch_size):
        #x = self.images[self._i: self._i+batch_size]
        #y = self.labels[self._i: self._i+batch_size]
        #self._i = (self._i + batch_size) % len(self.images)
        n = len(self.images)
        rand = [np.random.choice(n) for i in range(batch_size)]
        x = self.images[rand]
        y = self.labels[rand]


        return x, y


class CifarDataManager(object):
    def __init__(self):
        self.train = CifarLoader(['data_batch_{}'.format(i) for i in range(1,6)]).load()
        self.test = CifarLoader(['test_batch']).load()

cifar = CifarDataManager()

print('number of train image : {}'.format(len(cifar.train.images)))
print('number of train labels : {}'.format(len(cifar.train.labels)))
print('number of test image : {}'.format(len(cifar.test.images)))
print('number of test labels : {}'.format(len(cifar.test.labels)))

#images = cifar.train.image
#display_cifar(images, 5)

init_weight = tf.initializers.truncated_normal(stddev=0.03)
init_bias = tf.initializers.constant(0.1)

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape = [None, 10])
rate = tf.placeholder(tf.float32)

w0 = tf.Variable(init_weight(shape=[3, 3, 3, 16], dtype=tf.float32))
b0 = tf.Variable(init_bias(shape=[16]), dtype = tf.float32)
conv0 = tf.nn.relu(tf.nn.conv2d(x, w0, strides=[1,1,1,1], padding='SAME') + b0)
conv0_bn = tf.layers.batch_normalization(conv0)
conv0_drop = tf.nn.dropout(conv0_bn, rate=rate)

w1 = tf.Variable(init_weight(shape=[3, 3, 16, 32], dtype=tf.float32))
b1 = tf.Variable(init_bias(shape=[32]), dtype = tf.float32)
conv1 = tf.nn.relu(tf.nn.conv2d(conv0_bn, w1, strides=[1,1,1,1], padding='SAME') + b1)
conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
conv1_bn = tf.layers.batch_normalization(conv1_pool)
conv1_drop = tf.nn.dropout(conv1_bn, rate=rate)

w2 = tf.Variable(init_weight(shape=[3, 3, 32, 64], dtype=tf.float32))
b2 = tf.Variable(init_bias(shape=[64]), dtype = tf.float32)
conv2 = tf.nn.relu(tf.nn.conv2d(conv1_pool, w2, strides=[1,1,1,1], padding='SAME') + b2)
conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
conv2_bn = tf.layers.batch_normalization(conv2_pool)
conv2_drop = tf.nn.dropout(conv2_bn, rate=rate)

w3 = tf.Variable(init_weight(shape=[5, 5, 64, 128], dtype=tf.float32))
b3 = tf.Variable(init_bias(shape=[128]), dtype = tf.float32)
conv3 = tf.nn.relu(tf.nn.conv2d(conv2_drop, w3, strides=[1,1,1,1], padding='SAME') + b3)
conv3_pool = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
conv3_bn = tf.layers.batch_normalization(conv3_pool)
conv3_drop = tf.nn.dropout(conv3_pool, rate=rate)


conv4_flat = tf.contrib.layers.flatten(conv3_drop)


w8 = tf.Variable(init_weight(shape=[4*4*128, 1024], dtype=tf.float32))
b8 = tf.Variable(init_bias(shape=[1024]), dtype = tf.float32)
full1 = tf.nn.relu(tf.matmul(conv4_flat, w8) + b8)
full1_drop = tf.nn.dropout(full1, rate=rate)
full1_bn = tf.layers.batch_normalization(full1_drop)

w9 = tf.Variable(init_weight(shape=[1024, 1024], dtype=tf.float32))
b9 = tf.Variable(init_bias(shape=[1024]), dtype = tf.float32)
full2 = tf.nn.relu(tf.matmul(full1_bn, w9) + b9)
full2_drop = tf.nn.dropout(full2, rate=rate)
full2_bn = tf.layers.batch_normalization(full2_drop)

w12 = tf.Variable(init_weight(shape=[1024, 10], dtype=tf.float32))
b12 = tf.Variable(init_bias(shape=[10]), dtype = tf.float32)
full5 = tf.matmul(full2_bn, w12) + b12


logits = full5

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
train = tf.train.MomentumOptimizer(0.005, 0.9, use_nesterov=True).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


x_test = cifar.test.images.reshape(10, 1000, 32, 32, 3)
y_test = cifar.test.labels.reshape(10, 1000, 10)

def test(sess):

    loss__ = np.mean([sess.run(loss, feed_dict={x:x_test[i], y:y_test[i], rate:0.0}) for i in range(10)])
    acc__ = np.mean([sess.run(accuracy, feed_dict={x: x_test[i], y: y_test[i], rate: 0.0}) for i in range(10)])

    print('test accuracy: {:.4}%'.format(acc__*100))
    print('test loss: {:.4}'.format(loss__ ))

    loss_test.append(loss__)
    accuracy_test.append(acc__)



TRAIN_SIZE = 50000
BATCH_SIZE = 128
STEPS = 300
EPOCH = 500

loss_epoch = []
accuracy_epoch = []
loss_test = []
accuracy_test = []
preacc = 0


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)

    for i in range(EPOCH):

        loss_batch = []
        accuracy_batch = []

        for j in range(STEPS):
            batch = cifar.train.next_batch(BATCH_SIZE)
            _, loss_, acc = sess.run([train, loss, accuracy], feed_dict={x:batch[0], y:batch[1], rate:0.5})
            loss_batch.append(loss_)
            accuracy_batch.append(acc)
            if (j+1)%10 == 0 :
                print('epoch: {}, steps: {}, train-loss: {}, train-accuracy: {}'.format(i+1, j+1, loss_, acc))
        test(sess)

        mean_loss = np.mean(loss_batch)
        mean_accuracy = np.mean(accuracy_batch)
        loss_epoch.append(mean_loss)
        accuracy_epoch.append(mean_accuracy)


        if preacc <  max(accuracy_test):
            save_epoch(sess, i+1)
            saver.save(sess, "./checkpoint/"+str(i+1)+".cpkt")

            value = max(accuracy_test)
            value_index = accuracy_test.index(value)
            preacc = value


            plt.title("epoch: {}, max-test-acc: {:.4}%, max-train-acc: {:.4}%".format(value_index+1, 100*value, 100*accuracy_epoch[value_index]))
            plt.plot(accuracy_epoch, 'r', label="accuracy_train")
            plt.plot(accuracy_test, 'b', label="accuracy_test")
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.legend(loc='lower right')
            plt.savefig("./checkpoint/accuracy.png")
            plt.cla()


            plt.title("Loss Graph")
            plt.plot(loss_epoch, 'r', label="loss_train")
            plt.plot(loss_test, 'b', label="loss_test")
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc='upper right')
            plt.savefig("./checkpoint/loss.png")
            plt.cla()

sess.close()





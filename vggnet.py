import tensorflow as tf
import os
import pickle
import numpy as np

CIFAR_DIR = "cifar-10-batches-py"

# tensorboard
# 1.指定面板图上显示的变量
# 2.训练过程中将这些变量计算出来，输出到文件中
# 3.文件解析 ./tensorboard --logdir=dir

# 利用pickle读取数据
def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='iso-8859-1')
        return data['data'], data['labels']

# Cifar数据处理类
class CifarData:
    def __init__(self, filenames, need_shuffle):
        all_data = []
        all_labels = []
        for filename in filenames:
            data, labels = load_data(filename)
            for item, label in zip(data, labels):
                all_data.append(item)
                all_labels.append(label)
        self._data = np.vstack(all_data)
        # 数据归一化（？
        # self._data = self._data / 127.5 - 1
        self._labels = np.hstack(all_labels)
        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()

    # 讲训练数据打乱 避免过拟合
    def _shuffle_data(self):
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("have no more examples")
        if end_indicator > self._num_examples:
            raise Exception("batch size is larger than all examples")
        batch_data = self._data[self._indicator: end_indicator]
        batch_labels = self._labels[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels

# batch_size
batch_size = 40

x = tf.placeholder(tf.float32, [batch_size, 3072])
y = tf.placeholder(tf.int64, [batch_size])
is_training =tf.placeholder(tf.bool, [])

x_image = tf.reshape(x, [-1, 3, 32, 32])
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])
x_image_list = tf.split(x_image, num_or_size_splits=batch_size, axis=0)

result_x_image_list = []
for x_single_image in x_image_list:
    x_single_image = tf.reshape(x_single_image, [32, 32, 3])
    data_aug_1 = tf.image.random_flip_left_right(x_single_image)
    data_aug_2 = tf.image.random_brightness(data_aug_1, max_delta=63)
    data_aug_3 = tf.image.random_contrast(data_aug_2, lower=0.2, upper=1.8)
    data_aug_3 = tf.reshape(data_aug_3, [1, 32, 32, 3])
    result_x_image_list.append(data_aug_3)
result_x_images = tf.concat(result_x_image_list, axis= 0)

result_x_image_normal = result_x_images /127.5 -1

def Myconv2d(input,name,is_training,output_channel= 32,kernel_size=(3,3),padding='same',activation= tf.nn.relu):
    conv_result = tf.layers.conv2d(input, output_channel, kernel_size, padding=padding, activation=None, name=name)
    batch_normalization_result = tf.layers.batch_normalization(conv_result, training = is_training)
    activation_result = activation(batch_normalization_result)
    return activation_result

def Mypooling2d(input, name):
    return tf.layers.max_pooling2d(input, (2,2), (2,2), name = name)

# conv1 :神经元图， feature_map 输出图像
conv1_1 = Myconv2d(result_x_image_normal, 'conv1_1', is_training)
conv1_2 = Myconv2d(conv1_1, 'conv1_2', is_training)
conv1_3 = Myconv2d(conv1_2, 'conv1_3', is_training)

# pooling1 16 * 16
pooling1 = Mypooling2d(conv1_3, 'pool1')

conv2_1 = Myconv2d(pooling1, 'conv2_1', is_training)
conv2_2 = Myconv2d(conv2_1, 'conv2_2', is_training)
conv2_3 = Myconv2d(conv2_2, 'conv2_3', is_training)
# 8 * 8
pooling2 = Mypooling2d(conv2_3, 'pool2')

conv3_1 = Myconv2d(pooling2,'conv3_1', is_training)
conv3_2 = Myconv2d(conv3_1, 'conv3_2', is_training)
conv3_3 = Myconv2d(conv3_2, 'conv3_3', is_training)
# 4 * 4
pooling3 = Mypooling2d(conv3_3, 'pool3')

# flatten [None , 4 * 4 * 32] 全连接层
flatten = tf.layers.flatten(pooling3)

y_ = tf.layers.dense(flatten, 10)

loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
predict = tf.math.argmax(y_, 1)

correct_prediction = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

with tf.name_scope('train_op'):
    train_op = tf.train.GradientDescentOptimizer(0.01
).minimize(loss)

# # tensorboard 日志
# loss_summary = tf.summary.scalar('loss', loss)
# accuracy_summary = tf.summary.scalar('accuracy', accuracy)
# source_image = (x_image + 1) * 127.5
# inputs_summary =tf.summary.image('inputs_image', x_image)

# merged_summary = tf.summary.merge_all()

# merged_summary_test = tf.summary.merge([loss_summary, accuracy_summary])

train_filenames = []
for i in range(1, 6):
    train_filenames.append(os.path.join(CIFAR_DIR, 'data_batch_%d' % i))

test_filenames = [os.path.join(CIFAR_DIR, 'test_batch')]

train_data = CifarData(train_filenames, True)
test_data = CifarData(test_filenames, False)
init = tf.global_variables_initializer()

train_steps = 10000
test_steps = 50

with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):
        batch_data, batch_labels = train_data.next_batch(batch_size)
        loss_val, acc_val, _ = sess.run(
            [loss, accuracy, train_op],
            feed_dict={
                x: batch_data,
                y: batch_labels,
                is_training: True
            }
        )
        if (i + 1) % 50 == 0:
            print('[Train] Step : %d, loss %4.5f, acc: %4.5f' % (i, loss_val, acc_val))
        all_test_acc_val = []
        if (i + 1) % 500 == 0:
            test_batch_data, test_batch_labels = test_data.next_batch(batch_size)
            test_acc_val = sess.run(
                [accuracy],
                feed_dict={
                    x: test_batch_data,
                    y: test_batch_labels,
                    is_training: False
                }
            )
            all_test_acc_val.append(test_acc_val)
            test_acc = np.mean(all_test_acc_val)
            print('[Test] Step: %d, acc: %4.5f' % (i + 1, test_acc))

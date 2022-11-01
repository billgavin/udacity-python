
# coding: utf-8

# In[ ]:

from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import problem_unittests as tests
import tarfile

cifar10_dataset_folder_path = 'cifar-10-batches-py'
floyd_cifar10_location = '/input/cifar-10/python.tar.gz'

if isfile(floyd_cifar10_location):
    tar_gz_path = floyd_cifar10_location
else:
    tar_gz_path = 'cifar-10-python.tar.gz'
    
class DLProgress(tqdm):
    last_block = 0
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num -self.last_block) * block_size)
        self.last_block = block_num
        
if not isfile(tar_gz_path):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', tar_gz_path, pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open(tar_gz_path) as tar:
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar)
        tar.close()

tests.test_folder_path(cifar10_dataset_folder_path)


# In[ ]:

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

import helper
import numpy as np

batch_id = 1
sample_id = 5
helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)


# In[ ]:

def normalize(x):
    A, X, Y, Z = x.shape
    result = np.empty([A, X, Y, Z])
    img_max = x.max()
    img_min = x.min()
    for a in range(A):
        for i in range(X):
            for j in range(Y):
                for k in range(Z):
                    result[a][i][j][k] = (x[a][i][j][k] - img_min) / (img_max - img_min)
    return result

tests.test_normalize(normalize)                


# In[ ]:

def one_hot_encode(x):
    labels = np.array(x)
    one_hot_labels = []
    for num in labels:
        one_hot = [0.0] * 10
        one_hot[num] = 1.0
        one_hot_labels.append(one_hot)
    return np.array(one_hot_labels)

tests.test_one_hot_encode(one_hot_encode)


# In[ ]:

# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)


# In[1]:

'''
DON'T MODIFY ANYTHING IN THIS CELL
'''

import pickle
import problem_unittests as tests
import helper

valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))


# ## 输入
# 
# ### 神经网络需要读取图片数据、one-hot 编码标签和丢弃保留概率（dropout keep probability）。请实现以下函数：
# 
# - 实现 neural_net_image_input
#     - 返回 TF Placeholder
#     - 使用 image_shape 设置形状，部分大小设为 None
#     - 使用 TF Placeholder 中的 TensorFlow name 参数对 TensorFlow 占位符 "x" 命名
# - 实现 neural_net_label_input
#     - 返回 TF Placeholder
#     - 使用 n_classes 设置形状，部分大小设为 None
#     - 使用 TF Placeholder 中的 TensorFlow name 参数对 TensorFlow 占位符 "y" 命名
# - 实现 neural_net_keep_prob_input
#     - 返回 TF Placeholder，用于丢弃保留概率
#     - 使用 TF Placeholder 中的 TensorFlow name 参数对 TensorFlow 占位符 "keep_prob" 命名
#     
# 这些名称将在项目结束时，用于加载保存的模型。
# 
# 注意：TensorFlow 中的 None 表示形状可以是动态大小。

# In[2]:

import tensorflow as tf
def neural_net_image_input(image_shape):
    a, b, c = image_shape
    nn_inputs_x = tf.placeholder(tf.float32, shape=[None, a, b, c], name='x')
    return nn_inputs_x

def neural_net_label_input(n_classes):
    nn_inputs_y = tf.placeholder(tf.float32, shape=[None, n_classes], name='y')
    return nn_inputs_y

def neural_net_keep_prob_input():
    return tf.placeholder(tf.float32, name='keep_prob')

tf.reset_default_graph()
tests.test_nn_image_inputs(neural_net_image_input)
tests.test_nn_label_inputs(neural_net_label_input)
tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)


# ## 卷积和最大池化层
# 
# 卷积层级适合处理图片.实现函数conv2d_maxpool:
# 
# - 使用conv_ksize, conv_num_outputs 和 x_tensor 的形状创建权重和偏置
# - 使用权重和conv_strides 对 x_tensor 应用卷积.
#     - 建议使用padding
# - 添加偏置
# - 向卷积中添加非线性激活 nonlinear avtivation
# - 使用 pool_ksize 和 pool_strides 应用最大池化
#     - 建议使用padding

# In[3]:

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    shape = [conv_ksize[0], conv_ksize[1], x_tensor.get_shape().as_list()[3], conv_num_outputs]
    weight_inital = tf.truncated_normal(shape, stddev=0.1)
    bias_inital = tf.constant(0.1, shape=[conv_num_outputs])
    weight = tf.Variable(weight_inital)
    bias = tf.Variable(bias_inital)
    conv_layer = tf.nn.conv2d(x_tensor, weight, strides=[1, conv_strides[0], conv_strides[1], 1], padding='VALID')
    conv_layer = tf.nn.bias_add(conv_layer, bias)
    conv_layer = tf.nn.relu(conv_layer)
    
    pool_layer = tf.nn.max_pool(conv_layer, ksize=[1, pool_ksize[0], pool_ksize[1], 1], strides=[1, pool_strides[0], pool_strides[1], 1], padding='VALID')
    return pool_layer

tests.test_con_pool(conv2d_maxpool)


# In[4]:

def flatten(x_tensor):
    shape = x_tensor.get_shape().as_list()
    return tf.reshape(x_tensor, [-1, shape[1] * shape[2] * shape[3]])

tests.test_flatten(flatten)


# In[5]:

def fully_conn(x_tensor, num_outputs):
    shape = [x_tensor.get_shape().as_list()[1], num_outputs]
    weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[num_outputs]))
    fc = tf.add(tf.matmul(x_tensor, weight), bias)
    fc = tf.nn.relu(fc)
    return fc

tests.test_fully_conn(fully_conn)


# In[6]:

def output(x_tensor, num_outputs):
    shape = [x_tensor.get_shape().as_list()[1], num_outputs]
    weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[num_outputs]))
    output = tf.add(tf.matmul(x_tensor, weight),bias)
    output = tf.nn.softmax(output)
    return output

tests.test_output(output)


# In[11]:

def conv_net(x, keep_prob):
    conv_pool1 = conv2d_maxpool(x, 8, (5, 5), (4, 4), (2, 2), (2, 2))
    conv_pool2 = conv2d_maxpool(conv_pool1, 16, (1, 1), (1, 1), (1, 1), (1, 1))
    conv_pool = tf.nn.dropout(conv_pool2, keep_prob)
    flat = flatten(conv_pool)
    fc = fully_conn(flat, 256)
    outputs = output(fc, 10)
    return outputs

tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32,32,3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()
learning_rate = 0.0001

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

tests.test_conv_net(conv_net)


# In[12]:

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    session.run(tf.global_variables_initializer())
    session.run(optimizer, feed_dict={
        x: feature_batch,
        y: label_batch,
        keep_prob: keep_probability
    })
    
tests.test_train_nn(train_neural_network)


# In[13]:

def print_stats(session, feature_batch, label_batch, cost, accuracy):
    loss = session.run(cost, feed_dict = {
        x: feature_batch,
        y: label_batch,
        keep_prob: 1.    
    })
    valid_acc = sess.run(accuracy, feed_dict ={
        x: valid_features,
        y: valid_labels,
        keep_prob: 1.
    })
    print('Loss: {:>10.10f} Validation Accuracy: {:.10f}'.format(loss, valid_acc))


# In[14]:

import numpy as np

epochs = 20
batch_id = 1
batch_size = 64
#keep_probability = np.array([0.75])
keep_probability = 0.75

print('Check the Training on a Singal Batch... {}'.format(keep_probability))
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction=0.7
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_id, batch_size):
            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
        print('Epoch {:>2}, CIFAR-10 Batch {}: '.format(epoch + 1, batch_id), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)


# In[ ]:

save_model_path = './image_classification'
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}: '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
    saver = tf.train.Saver()
    saver_path = saver.save(sess, save_model_path)


# In[ ]:

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

import tensorflow as tf
import pickle
import helper
import random

try:
    if batch_size:
        pass
except NameError:
    batch_size = 64
    
n_samples = 4
top_n_predictions = 3

def test_model():
    test_features, test_labels = pickle.load(open('preprocess_test.p', mode='rb'))
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph, config=config) as sess:
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)
        
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        test_batch_acc_total = 0
        test_batch_count = 0
        
        for test_feature_batch, test_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict = {
                    loaded_x: test_feature_batch,
                    loaded_y: test_label_batch,
                    loaded_keep_prob: 1.0
                })
            test_batch_count += 1
        print('Testing Accuracy: {}\n'.format(test_batch_acc_total / test_batch_count))
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict = {
                loaded_x: random_test_features,
                loaded_y: random_test_labels,
                loaded_keep_prob: 1.0
            })
        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)
test_model()


# In[ ]:




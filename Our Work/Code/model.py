import tensorflow as tf
from tensorflow import keras
import numpy as np

def Generator(input_image) :
    
    with tf.compat.v1.variable_scope("generator") :
        # Convolution Layer
        W1 = weights_init([9, 9, 3, 64], name='W1')
        B1 = bias_init([64], name="B1")
        C1 = tf.nn.relu(conv2d(input_image, W1, B1))

        # Residual Block * 4
        C3 = Residual_Block(C1, 1)
        C5 = Residual_Block(C3, 2)
        C7 = Residual_Block(C5, 3)
        C9 = Residual_Block(C7, 4)

        # Convolution Layer * 2
        W10 = weights_init([3,3,64,64], name='W10')
        B10 = bias_init([64], name="B10")
        C10 = tf.nn.relu(conv2d(C9, W10, B10))

        W11 = weights_init([3,3,64,64], name='W11')
        B11 = bias_init([64], name="B11")
        C11 = tf.nn.relu(conv2d(C10, W11, B11))

        # Final Layer
        W12 = weights_init([9, 9, 64, 3], name="W12")
        B12 = bias_init([3], name="B12")
        C12 = tf.nn.tanh(conv2d(C11, W12, B12)) * 0.58 + 0.5

    return C12

def Discriminator(image) :
    
    with tf.compat.v1.variable_scope("discriminator") :
        conv1 = conv_layer(image, 48, 11, 4, batch_nn=False)
        conv2 = conv_layer(conv1, 128, 5, 2)
        conv3 = conv_layer(conv2, 192, 3, 1)
        conv4 = conv_layer(conv3, 192, 3, 1)
        conv5 = conv_layer(conv4, 128, 3, 2)

        # Flatten

        flat_size = 128 * 7 * 7
        conv5_flat = tf.reshape(conv5, [-1, flat_size])

        # FC Layer

        weights_fc = tf.Variable(tf.random.normal([flat_size, 1024], stddev=0.01))
        bias_fc = tf.Variable(tf.constant(0.01, shape=[1024]))
        fc = tf.nn.leaky_relu(tf.matmul(conv5_flat, weights_fc) + bias_fc, alpha=0.2)

        # FC Layer
        weights_out = tf.Variable(tf.random.normal([1024, 2], stddev=0.01))
        bias_out = tf.Variable(tf.constant(0.01, shape=[2]))

        output = tf.nn.softmax(tf.matmul(fc, weights_out)+bias_out)

    return output

def conv_layer(input_to_layer, num_filters, kernel_size, stride, padding = 'same', batch_nn = True) :
    
    stride_array = [1, stride, stride, 1]
    filter_init = init_weights_matrix(input_to_layer, num_filters, kernel_size)
    bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))
    
    output = tf.nn.conv2d(input_to_layer, filter_init, stride_array, padding='SAME') + bias
    output = tf.nn.leaky_relu(output, alpha=0.2)
    
    if batch_nn :
        output = batch_normalization(output)
    
    return output

def init_weights_matrix(input_to_layer, out_channels, kernel_size) :
    
    _, row, col, in_channels = [i for i in input_to_layer.get_shape()]
    kernel_shape = [kernel_size, kernel_size, in_channels, out_channels]
    
    kernel_init = tf.Variable(tf.random.normal(kernel_shape, stddev=0.01), dtype=tf.dtypes.float32)
    return kernel_init

def BatchNormalization(input_to_layer) :
    batch_size, rows, cols, channels = [i for i in input_to_layer.get_shape()]
    
    mean, variance = tf.nn.moments(input_to_layer, [1,2], keepdims=True)
    shift = tf.Variable(tf.zeros([channels]))
    scale = tf.Variable(tf.ones([channels]))
    
    epsilon = 1e-3
    normalized = ( input_to_layer - mean ) / ( ( variance + epsilon ) ** ( 0.5 ))
    
    return scale * normalized + shift

def Residual_Block(input_to_layer, residual_layer_no) :
    
    W1 = weights_init([3,3,64,64], name="W"+str(residual_layer_no*2))
    B1 = bias_init([64], name="B"+str(residual_layer_no*2))
    C1 = tf.nn.relu(BatchNormalization(conv2d(input_to_layer, W1, B1)))
    
    W2 = weights_init([3,3,64,64], name="W"+str(residual_layer_no*2+1))
    B2 = bias_init([64], name="B"+str(residual_layer_no*2+1))
    C2 = tf.nn.relu(BatchNormalization(conv2d(C1, W2, B2))) + input_to_layer
    
    return C2

def weights_init(shape, name) :
    return tf.Variable(tf.random.normal(shape, stddev=0.01), name=name)

def bias_init(shape, name) :
    return tf.Variable(tf.constant(0.01, shape), name=name)

def conv2d(x, W, B) :
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')+B
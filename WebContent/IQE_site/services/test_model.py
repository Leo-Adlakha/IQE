# python test_model.py model=<model> resolution=<resolution> use_gpu=<use_gpu>
# <model> = {iphone, blackberry, sony}
# <resolution> = {orig, high, medium, small, tiny}
# <use_gpu> = {true, false}
# example:  python test_model.py model=iphone resolution=orig use_gpu=true

from scipy import misc
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import sys

def resnet(image):

    with tf.variable_scope("generator"):

        # Convolutional layer

        W1 = weight_variable([9, 9, 3, 64], name="W1");
        b1 = bias_variable([64], name="b1");
        c1 = tf.nn.relu(conv2d(image, W1) + b1)

        # Residual layer 1

        W2 = weight_variable([3, 3, 64, 64], name="W2");
        b2 = bias_variable([64], name="b2");
        c2 = tf.nn.relu(_instance_norm(conv2d(c1, W2) + b2))

        W3 = weight_variable([3, 3, 64, 64], name="W3");
        b3 = bias_variable([64], name="b3");
        c3 = tf.nn.relu(_instance_norm(conv2d(c2, W3) + b3)) + c1

        # Residual layer 2

        W4 = weight_variable([3, 3, 64, 64], name="W4");
        b4 = bias_variable([64], name="b4");
        c4 = tf.nn.relu(_instance_norm(conv2d(c3, W4) + b4))

        W5 = weight_variable([3, 3, 64, 64], name="W5");
        b5 = bias_variable([64], name="b5");
        c5 = tf.nn.relu(_instance_norm(conv2d(c4, W5) + b5)) + c3

        # Residual layer 3

        W6 = weight_variable([3, 3, 64, 64], name="W6");
        b6 = bias_variable([64], name="b6");
        c6 = tf.nn.relu(_instance_norm(conv2d(c5, W6) + b6))

        W7 = weight_variable([3, 3, 64, 64], name="W7");
        b7 = bias_variable([64], name="b7");
        c7 = tf.nn.relu(_instance_norm(conv2d(c6, W7) + b7)) + c5

        # Residual layer 4

        W8 = weight_variable([3, 3, 64, 64], name="W8");
        b8 = bias_variable([64], name="b8");
        c8 = tf.nn.relu(_instance_norm(conv2d(c7, W8) + b8))

        W9 = weight_variable([3, 3, 64, 64], name="W9");
        b9 = bias_variable([64], name="b9");
        c9 = tf.nn.relu(_instance_norm(conv2d(c8, W9) + b9)) + c7

        # Convolutional layer

        W10 = weight_variable([3, 3, 64, 64], name="W10");
        b10 = bias_variable([64], name="b10");
        c10 = tf.nn.relu(conv2d(c9, W10) + b10)

        # Convolutional layer

        W11 = weight_variable([3, 3, 64, 64], name="W11");
        b11 = bias_variable([64], name="b11");
        c11 = tf.nn.relu(conv2d(c10, W11) + b11)

        # Output layer

        W12 = weight_variable([9, 9, 64, 3], name="W12");
        b12 = bias_variable([3], name="b12");
        prediction = tf.nn.tanh(conv2d(c11, W12) + b12) * 0.58 + 0.5

    return prediction

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def _instance_norm(net):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net - mu) / (sigma_sq + epsilon) ** (.5)
    return scale * normalized + shift

def get_resolutions():

    # IMAGE_HEIGHT, IMAGE_WIDTH

    res_sizes = {}

    res_sizes["iphone"] = [1536, 2048]
    res_sizes["blackberry"] = [1560, 2080]
    res_sizes["sony"] = [1944, 2592]
    res_sizes["high"] = [1260, 1680]
    res_sizes["medium"] = [1024, 1366]
    res_sizes["small"] = [768, 1024]
    res_sizes["tiny"] = [600, 800]

    return res_sizes

def get_specified_res(res_sizes, phone, resolution):

    if resolution == "orig":
        IMAGE_HEIGHT = res_sizes[phone][0]
        IMAGE_WIDTH = res_sizes[phone][1]
    else:
        IMAGE_HEIGHT = res_sizes[resolution][0]
        IMAGE_WIDTH = res_sizes[resolution][1]

    IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3

    return IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE

def extract_crop(image, resolution, phone, res_sizes):

    if resolution == "orig":
        return image

    else:

        x_up = int((res_sizes[phone][1] - res_sizes[resolution][1]) / 2)
        y_up = int((res_sizes[phone][0] - res_sizes[resolution][0]) / 2)

        x_down = x_up + res_sizes[resolution][1]
        y_down = y_up + res_sizes[resolution][0]

        return image[y_up : y_down, x_up : x_down, :]

def predict(image_path, save_path, models_path) :

    phone = "iphone"
    resolution = "orig"
    use_gpu = "false"

    # get all available image resolutions
    res_sizes = get_resolutions()

    # get the specified image resolution
    IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE = get_specified_res(res_sizes, phone, resolution)

    # disable gpu if specified
    config = tf.ConfigProto(device_count={'GPU': 0}) if use_gpu == "false" else None

    # create placeholders for input images
    x_ = tf.placeholder(tf.float32, [1, IMAGE_SIZE])
    x_image = tf.reshape(x_, [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    # generate enhanced image
    enhanced = resnet(x_image)

    with tf.Session(config=config) as sess:

        tf.Graph()
        # load pre-trained model
        saver = tf.train.Saver()
        saver.restore(sess, models_path + phone)

        # load training image and crop it if necessary

        print("Processing image")
        # image = np.float16(np.array(Image.fromarray()))

        image = np.float16(misc.imresize(misc.imread(image_path), res_sizes[phone])) / 255

        image_crop = extract_crop(image, resolution, phone, res_sizes)
        image_crop_2d = np.reshape(image_crop, [1, IMAGE_SIZE])

        # get enhanced image

        enhanced_2d = sess.run(enhanced, feed_dict={x_: image_crop_2d})
        enhanced_image = np.reshape(enhanced_2d, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])

        # save the results as .png images
        print("Saving File")

        misc.imsave(save_path, enhanced_image)



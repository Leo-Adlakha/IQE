import numpy as np
import imageio
import rawpy
import matplotlib.pyplot as plt
import os
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import save_img
import scipy.stats as st
import sys
from functools import reduce
from operator import mul

def prepare_training_data_from_s7(dir_path='/Users/leoadlakha/Documents/Research Work/Image Qualty Enhancement/Dataset/samsungs7Dataset/S7-ISP-Dataset/') :
    
    '''
    This function prepares the s7 Dataset in the form of 
    100 * 100 * 3 for each image
    
    dir_path - The path to the s7 Directory
    '''
    
    X_s7, Y_s7 = utils.load_samsung_dataset(dir_path='/Users/leoadlakha/Documents/Research Work/Image Qualty Enhancement/Dataset/samsungs7Dataset/S7-ISP-Dataset/')
    X = []
    Y = []
    c = 0
    for i, j in zip(X_s7, Y_s7) :
        c = c + 1
        print(c)
        crops_X = np.array(utils.crop_image_without_padding(100, 100, i)).astype('int32')
        crops_Y = np.array(utils.crop_image_without_padding(100, 100, j)).astype('int32')
        X.append(crops_X)
        Y.append(crops_Y)
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    print("Ready: Training Data from s7 Dataset")
    return X, Y

def prepare_training_data_from_UFO120(dir_path='/Users/leoadlakha/Documents/Research Work/Image Qualty Enhancement/Dataset/UFO-120/train_val'):
    
    '''
    This function prepares the UFO120 Dataset in the form of 
    100 * 100 * 3 for each image
    
    dir_path - The path to the UFO120 Directory
    '''
    
    X_UFO, Y_UFO = utils.load_UFO120_Dataset(dir_path='/Users/leoadlakha/Documents/Research Work/Image Qualty Enhancement/Dataset/UFO-120/train_val')
    X = []
    Y = []
    c = 0
    for i, j in zip(X_UFO, Y_UFO) :
        c = c + 1
        print(c)
        crops_X = np.array(utils.crop_image_without_padding(100, 100, i)).astype('int32')
        crops_Y = np.array(utils.crop_image_without_padding(100, 100, j)).astype('int32')
        X.append(crops_X)
        Y.append(crops_Y)
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    return X, Y

def rotate_image_samsung_dataset(X, Y, idx=[26, 46, 68, 74]) :
    for i in idx :
        Y[i] = np.rot90(Y[i], k=1)
        X[i] = np.rot90(X[i], k=1)

def load_samsung_dataset(dir_path = '/Users/leoadlakha/Documents/Research Work/Image Quality Enhancement/Dataset/samsungs7Dataset/S7-ISP-Dataset/') :
    
    '''
    This function takes the path of the directory, and return the images read
    in .jpg format and returns them as a numpy array.
    
        dir_path - It is the path of Samsung-s7 Directory.
    '''
    
    X = []
    Y = []
    for folder in os.listdir(dir_path) :
        folder_path = dir_path + folder
        if not folder.startswith('2016') :
            continue
        for file in os.listdir(folder_path) :
            if file.endswith('.jpg') :
                img_path = os.path.join(folder_path, file)
                image = Image.open(img_path)
                if ( file.startswith('medium') and file.endswith('.jpg') ) :
                    Y.append(np.asarray(image))
                elif ( file.startswith('short') and file.endswith('.jpg') ) :
                    X.append(np.asarray(image))
                        
    X = np.array(X)
    Y = np.array(Y)
    print("Loaded Samsung Dataset")
#     rotate_image_samsung_dataset(X, Y, idx=[26, 46, 68, 74])
    
    return X, Y

def load_UFO120_Dataset(dir_path='/Users/leoadlakha/Documents/Research Work/Image Qualty Enhancement/Dataset/UFO-120/train_val') :
    
    '''
    This function takes the path of the directory, and return the images read
    in .jpg format and returns them as a numpy array.
    
        dir_path - It is the path of UFO120 Directory.
    '''
    
    x_train = os.path.join(dir_path, 'lrd')
    y_train = os.path.join(dir_path, 'hr')
    
    X = []
    Y = []
    for x, y in zip(os.listdir(x_train), os.listdir(y_train)) :
        x_path = os.path.join(x_train, x)
        y_path = os.path.join(y_train, y)

        x_image = Image.open(x_path)
        y_image = Image.open(y_path)
        y_image = y_image.resize(x_image.size)

        x_data = np.asarray(x_image)
        y_data = np.asarray(y_image)

        X.append(x_data)
        Y.append(y_data)
    X = np.array(X)
    Y = np.array(Y)
    print("Loaded UFO120 Dataset")
    
    return X, Y

def get_Data_For_DPED_Model(x, y):
    
    '''
    This function takes the path of the directories of both LQ and HQ Images, 
    and return the images read in .jpg format and returns them as a numpy array.
    
        x - It is the LQ Images Directory taken by Blueberry/Iphone/Sony Camera.
        y - It is the HQ Images Directory taken by Canon Camera.
    '''
    
    X = []
    Y = []
    for i, j in zip(os.listdir(x), os.listdir(y)) :

        if i.endswith('.jpg') and j.endswith('.jpg') :
            x_image = Image.open(os.path.join(x, i))
            y_image = Image.open(os.path.join(y, j))

            x_array = np.asarray(x_image)
            y_array = np.asarray(y_image)

            X.append(x_array)
            Y.append(y_array)

    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y

def load_DPED_Dataset(dir_path='/Users/leoadlakha/Documents/Research Work/Image Qualty Enhancement/Dataset/dped') :
    
    '''
    This function takes the path of the directory, and return the images read
    in .jpg format and returns them as a numpy array.
    
        dir_path - It is the path of DPED Directory.
    '''
    
    bb = os.path.join(dir_path, 'blackberry/training_data')
    ip = os.path.join(dir_path, 'iphone/training_data')
    sn = os.path.join(dir_path, 'sony/training_data')
    
    x_bb = os.path.join(bb, 'blackberry')
    x_ip = os.path.join(ip, 'iphone')
    x_sn = os.path.join(sn, 'sony')

    y_bb = os.path.join(bb, 'canon')
    y_ip = os.path.join(ip, 'canon')
    y_sn = os.path.join(sn, 'canon')
    
    X_BB, Y_BB = get_Data_For_DPED_Model(x_bb, y_bb)
    print("Done with Blueberry")
    X_IP, Y_IP = get_Data_For_DPED_Model(x_ip, y_ip)
    print("Done with Iphone")
    X_SN, Y_SN = get_Data_For_DPED_Model(x_sn, y_sn)
    print("Done with Sony")
    
    return np.concatenate((X_BB, X_IP, X_SN)), np.concatenate((Y_BB, Y_IP, Y_SN))

def show_img(x) :
    
    '''
    This functions plots the given image using matplotlib library
    
        x - It is the np.array() for the image
    '''
    
    plt.imshow(x)
    plt.title(str(x.shape))
    plt.show()


def crop_image_without_padding(x, y, img_array):

    '''
    This Function return a list of crope images in form numpy array 
    without adding padding rejecting the left alone one

    x, y - dimension of crop image
    img_array - array representing image
    '''

    li = []
    for i in range(x, img_array.shape[0] + 1, x):
        for j in range(y, img_array.shape[1] + 1, y):
            temp = np.zeros((x,y,3), dtype='float32')
            for l in range(i-x, i, 1):
                temp[l-i+x][:][:] = img_array[l][j-y: j][:]
            li.append(temp);
    li = np.array(li)
    return li


def crop_image_with_padding(x, y, img_array):

    '''
    This Function return a list of crope images in form numpy array 
    with padding as part of image

    x, y - dimension of crop image
    img_array - array representing image
    '''

    li = []
    for i in range(0, img_array.shape[0], x):
        if(i+x > img_array.shape[0]):
            i = img_array.shape[0] - x
        for j in range(0, img_array.shape[1], y):
            if(j+y > img_array.shape[1]):
                j = img_array.shape[1] - y
            temp = np.zeros((x,y,3), dtype='float32')
            for l in range(i, i+x, 1):
                temp[l-i][:][:] = img_array[l][j: j+y][:]
            li.append(temp);
    return li

def log10(x):
    
    '''
    Function to calculate log10(x)
    
    x - Value for which log10 is to be returned
    '''
    
    numerator = tf.compat.v1.log(x)
    denominator = tf.compat.v1.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def _tensor_size(tensor):
    
    '''
    Returns the Size of the Tensor
    
    tensor - A Tensor
    '''
    
    return reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    
    '''
    Function to Help the blur Function which helps in calculating 
    the Color Loss of the model.
    '''
    
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter


def blur(x):
    
    '''
    Helper function used for calculating the Color Loss
    '''
    
    kernel_var = gauss_kernel(21, 3, 3)
    return tf.nn.depthwise_conv2d(x, kernel_var, [1, 1, 1, 1], padding='SAME')

# def combine_crop(li, s, x, y):

# 	'''
# 	This Function combine the crop images into the original one

# 	x, y - dimension of crop image
# 	li - list of images in form of numpy array
# 	s - shape of original image 
# 	'''

# 	temp = np.zeros((s), dtype='float32')
# 	prev_idx = 0
# 	for i in range(s[0]):
# 		idx = prev_idx
# 		if (i%x == 0 and i > 0):
# 			idx += (s[1]//y)
# 			prev_idx = idx
# 		for j in range(s[1]):
# 			if(j%y == 0):
# 				arr = li[idx]
# 				idx += 1
# 			temp[i][j][:] = arr[i%x][j%y][:]
# 	return temp
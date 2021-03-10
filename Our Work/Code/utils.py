import numpy as np
import imageio
import rawpy
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import save_img

def load_samsung_dataset(dir_path = '/Users/leoadlakha/Documents/Research Work/Image Quality Enhancement/Dataset/samsungs7Dataset/S7-ISP-Dataset/') :
    
    '''
    This function takes the path of the directory, and return the images read
    in .dng format and returns them as a numpy array.
    
        dir_path - It is the path of Samsung-s7 Directory.
    '''
    
    X = np.array()
    Y = np.array()
    c = 0
    for folder in os.listdir(dir_path) :
        folder_path = dir_path + folder
        c += 1 
        print(c)
        if not folder.startswith('2016') :
            continue
        for file in os.listdir(folder_path) :
            if file.endswith('.dng') :
                img_path = folder_path + '/' + file
                with rawpy.imread(img_path) as raw :
                    if file.startswith('medium') :
                        Y.append(raw.postprocess())
                    else :
                        X.append(raw.postprocess())
                        
    return X, Y

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
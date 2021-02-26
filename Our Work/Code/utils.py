import numpy as np
import imageio
import rawpy
import matplotlib.pyplot as plt
import os

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
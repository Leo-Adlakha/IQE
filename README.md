# IQE

Leo Adlakha, Prateek Bhardwaj, Abhijeet Singh Varun

Netaji Subhas University of Technology, Delhi, India

[IQE](https://iqe-os.herokuapp.com/): Image Quality Enhancement is developed to enhance the quality of low-light images with size (1536, 2048, 3). Some of the results are shown below: 

<p align="center">
    <img src="https://github.com/Leo-Adlakha/IQE/images/result.png" height="256" width="341">
    <img src="https://github.com/Leo-Adlakha/IQE/images/result1.png" height="256" width="341">
    <img src="https://github.com/Leo-Adlakha/IQE/images/result2.png" height="256" width="341">
</p>

## Requirements

_requirements.txt_ contains the Python packages used by the code and the website.

## Abstract

<p>
    We present a novel approach to adjust various image properties of low-light images to yield an enhanced image with better contrast, brightness, etc. The problem is to enhance the quality of images taken from various mobile devices like iPhone, Samsung etc. We propose a deep learning-based approach involving Deep Convolutional Generative Adversarial Networks (GANs), that can be trained with a pair of images of low quality as a noise matrix and the output of the generator is used to compare the results when this output and high quality image ( groundtruth ) are fed into the Discriminator to obtain various losses involving color, texture, etc. It works very well on real life test images taken from the publicly available datasets like samsung-s7 and MIT-Adobe 5K Dataset.
</p>

## Technologies Used

1. Tensorflow
2. Scipy
3. PIL
4. Django

## Datasets Used

1. MIT-Adobe 5K Dataset
2. Samsung-s7 Dataset

## Installation

### For Model

1. Clone the Github Repository

```
$ git clone https://github.com/Leo-Adlakha/IQE && cd IQE/
```

2. Create a new Environment for the Project using virtualenv

**Note** - Install pip package manager from [here](https://pip.pypa.io/en/stable/installing/)

```
$ pip install virtualenv
$ python3 -m venv env
$ source env/bin/activate
```


3. Install all the necessary Packages in requirements.txt using pip package manager

```
$ pip install -r requirement.txt
```
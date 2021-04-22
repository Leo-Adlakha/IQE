# IQE: Image Quality Enhancement using DCGANs

*Leo Adlakha, Prateek Bhardwaj, Abhijeet Singh Varun*
*from Netaji Subhas University of Technology, Delhi, India*

**[IQE](https://iqe-os.herokuapp.com/)**: Image Quality Enhancement is developed to enhance the quality of low-light images with size (1536, 2048, 3). Some of the results are shown below: 

![demo](images/result.png)
![demo](images/result1.png)
![demo](images/result2.png)

## Requirements

_requirements.txt_ contains the Python packages used by the code and the website.

## Abstract

<p>
    We present a novel approach to adjust various image properties of low-light images to yield an enhanced image with better contrast, brightness etc. The problem is to enhance the quality of images taken from various mobile devices like iPhone, Samsung etc. We propose a deep learning-based approach involving Deep Convolutional Generative Adversarial Networks (GANs), that can be trained with a pair of images of low quality as a noise matrix and the output of the generator is used to compare the results when this output and high quality image (groundtruth) are fed into the Discriminator to obtain various losses involving color, texture etc. It works very well on real life test images taken from the publicly available datasets like Samsung-S7 and MIT-Adobe 5K.
</p>

## Technologies Used

1. TensorFlow
2. Scipy
3. PIL
4. Django

## Datasets Used

1. MIT-Adobe 5K Dataset
2. Samsung-S7 Dataset

## Installation and Setup Guide for Unix/Linux

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
$ pip install -r requirements.txt
```

4. Change Directory to that of test_model.py

```
$ cd MachineLearningContent/Final\ Code/
```

5. Run the model
* If you have a gpu then put use_gpu to true else to false

For ex -

```
$ python3 test_model.py use_gpu=false
```

### For WebSite

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
$ pip install -r requirements.txt
```

4. Change Directory to run the Django Server

```
$ cd WebContent/IQE_site/
```

5. Run server on your machine

```
$ python3 manage.py runserver
```

### To Setup Environment in Anaconda (Windows)

Download and import [IQE_Project.yml](https://anaconda.org/bhardwajprateek291200/iqe_project).

Now, you may open your browser and go to http://127.0.0.1:8000/

After, you see the website, in the Enhancer Section, upload an image of dimension (1536, 2048, 3) and wait for the magic to happen :sparkles: !

Your result will be automatically downlaoded.

## Contact Us

[Leo Adlakha](mailto:leoa.co18@nsut.ac.in)

[Prateek Bhardwaj](mailto:prateekb.co18@nsut.ac.in)

[Abhijeet Singh Varun](mailto:abhijeets.co18@nsut.ac.in)

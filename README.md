# Human-Skin-Detection
Human Skin Detection Using RGB, HSV and YCbCr Color. this code is all algorithmic way to detect human skin and best for the preprocessing of the Deep learning as zero shot methods. This method is implemented from a paper!

<img src="/Pictures/Output/jensen_huang.jpg"/>

## Tech :hammer_and_wrench: Languages and Tools :

<div>
  <img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original.svg" title="Python" alt="Python" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/jupyter/jupyter-original.svg" title="Jupyter Notebook" alt="Jupyter Notebook" width="40" height="40"/>&nbsp;
  <img src="https://assets.st-note.com/img/1670632589167-x9aAV8lmnH.png" title="Google Colab" alt="Google Colab" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/opencv/opencv-original.svg" title="OpenCV" alt="OpenCV" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/numpy/numpy-original.svg" title="Numpy" alt="Numpy" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/matplotlib/matplotlib-original.svg"  title="MatPlotLib" alt="MatPlotLib" width="40" height="40"/>&nbsp;
</div>

- Python : Popular language for implementing Neural Network
- Jupyter Notebook : Best tool for running python cell by cell
- Google Colab : Best Space for running Jupyter Notebook with hosted server
- OpenCV : Best Library for working with images
- Numpy : Best Library for working with arrays in python
- MatPlotLib : Library for showing the charts in python

## Run the Notebook on Google Colab

You can easily run this code on google colab by just clicking this badge [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AsadiAhmad/Human-Skin-Detection/blob/main/Code/Human-Skin-Detector.ipynb)

## Tutorial

### Step 1: Import Libraries

we need to import these libraries :

`cv2`, `numpy`, `matplotlib`

```sh
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

### Step 2: Download Images

We need to Download the images from my `Github` repository or you can download other human pictures by your own.

```sh
!wget https://raw.githubusercontent.com/AsadiAhmad/Human-Skin-Detection/main/Pictures/Input/jensen_huang.jpg -O jensen_huang.jpg
!wget https://raw.githubusercontent.com/AsadiAhmad/Human-Skin-Detection/main/Pictures/Input/elon_musk.jpg -O elon_musk.jpg
!wget https://raw.githubusercontent.com/AsadiAhmad/Human-Skin-Detection/main/Pictures/Input/mark_zukerberg.jpg -O mark_zukerberg.jpg
!wget https://raw.githubusercontent.com/AsadiAhmad/Human-Skin-Detection/main/Pictures/Input/linus_torvalds.jpg -O linus_torvalds.jpg
```

### Step 3: Load Images

We need to load images into `python` variables we ues `OpenCV` library to read the images also the format of the images are `nd.array`.

```sh
bgr_image = cv2.imread('jensen_huang.jpg')
elon_musk = cv2.imread('elon_musk.jpg')
mark_zukerberg = cv2.imread('mark_zukerberg.jpg')
linus_torvalds = cv2.imread('linus_torvalds.jpg')
```

### Step 4: Extract HSV and YCrCb color formats

In this algorithmic approch we need to get teh HSV and YCrCb color formats for calculating the conditions

```sh
hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
ycrcb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
```

HSV color format :

<div display=flex align=center>
  <img src="/Pictures/Colors/HSV.jpg" width="400px"/>
</div>

YCbCr color format :

<div display=flex align=center>
  <img src="/Pictures/Colors/YCbCr.jpg" width="400px"/>
</div>

### Step 5: Invert the Binary Image

For using the morpholgy in image processing we need to invert the images 

```sh
inverted_image = 255 - binary_image
```

<div display=flex align=center>
  <img src="/Pictures/2.jpg" width="400px"/>
</div>

### Step 6: Opening Image for Completely remove Noise

Actually Opening have two section :

1- Erosion for removing noise that are not eleminated by midan filter and created after biniarization the image.

```sh
kernel = np.ones((2, 2), np.uint8)
erosion = cv.erode(inverted_image, kernel, iterations = 1)
```

<div display=flex align=center>
  <img src="/Pictures/3.jpg" width="400"/>
</div>

2- Dilation for bolding the text because after the erosion we lose some part of the text so we need to refill the text.

```sh
kernel2 = np.ones((5, 5), np.uint8)
dilation = cv.dilate(erosion, kernel2, iterations = 1)
```

<div display=flex align=center>
  <img src="/Pictures/4.jpg" width="400px"/>
</div>

### Step 7: Invert Image again

we have an Image with white text and black background and we don't want this so we invert that again.

```sh
inverted_image2 = 255 - dilation
```

<div display=flex align=center>
  <img src="/Pictures/5.jpg" width="400px"/>
</div>

### Step 8: All together for other images

so in this step we put all of things together and test that for other images.

```sh
def binarization_image(image, blur_value=5, kernel_erosion=(2, 2), kernel_dilation=(5, 5)):
    noise_removed = cv.medianBlur(image, blur_value)
    binary_image = cv.adaptiveThreshold(noise_removed, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
    inverted_image = 255 - binary_image
    kernel = np.ones(kernel_erosion, np.uint8)
    erosion = cv.erode(inverted_image, kernel, iterations = 1)
    kernel2 = np.ones(kernel_dilation, np.uint8)
    dilation = cv.dilate(erosion, kernel2, iterations = 1)
    inverted_image2 = 255 - dilation
    return inverted_image2
```

## License

This project is licensed under the MIT License.
